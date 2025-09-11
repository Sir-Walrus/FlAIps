//Module name: libbps
//Author: Alcaro
//Date: 2025-09-11 (rewritten for threading and SIMD)
//Licence: GPL v3.0 or higher

#include "libbps.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <thread>
#include <vector>
#include <atomic>
#include "crc32.h"

// SIMD helpers: AVX2 for x86, NEON for ARM, fallback to memcpy
inline void fast_copy(uint8_t* dst, const uint8_t* src, size_t len) {
#if defined(__AVX2__)
    // Use AVX2 for large blocks
    while (len >= 32) {
        __m256i temp = _mm256_loadu_si256((const __m256i*)src);
        _mm256_storeu_si256((__m256i*)dst, temp);
        dst += 32; src += 32; len -= 32;
    }
#elif defined(__ARM_NEON)
    while (len >= 16) {
        uint8x16_t temp = vld1q_u8(src);
        vst1q_u8(dst, temp);
        dst += 16; src += 16; len -= 16;
    }
#endif
    // Fallback
    if (len)
        memcpy(dst, src, len);
}

enum { SourceRead, TargetRead, SourceCopy, TargetCopy };

static bool decodenum(const uint8_t*& ptr, size_t& out) {
    out = 0;
    unsigned int shift = 0;
    while (true) {
        uint8_t next = *ptr++;
        size_t addthis = (next & 0x7F);
        if (shift) addthis++;
        addthis <<= shift;
        out += addthis;
        if (next & 0x80) break;
        shift += 7;
    }
    return true;
}

struct PatchJob {
    size_t out_offset;
    size_t in_offset; // only for SourceRead
    size_t patch_offset; // only for TargetRead
    size_t length;
    int action;
};

// Identify large jobs for threading (only SourceRead and TargetRead, >128kB)
static void batch_jobs(const std::vector<PatchJob>& jobs, std::vector<PatchJob>& threaded, std::vector<PatchJob>& single) {
    for (const PatchJob& job : jobs) {
        if ((job.action == SourceRead || job.action == TargetRead) && job.length >= 128*1024) {
            threaded.push_back(job);
        } else {
            single.push_back(job);
        }
    }
}

enum bpserror bps_apply(struct mem patch, struct mem in, struct mem * out, struct mem * metadata, bool accept_wrong_input)
{
    enum bpserror error = bps_ok;
    out->len = 0;
    out->ptr = nullptr;
    if (metadata) { metadata->len = 0; metadata->ptr = nullptr; }
    if (patch.len < 4+3+12) return bps_broken;

    const uint8_t *patchat = patch.ptr;
    const uint8_t *patchend = patch.ptr + patch.len - 12;

    if (patchat[0]!='B' || patchat[1]!='P' || patchat[2]!='S' || patchat[3]!='1') return bps_broken;
    patchat += 4;

    uint32_t crc_in_e = *(uint32_t*)(patch.ptr+patch.len-12);
    uint32_t crc_out_e = *(uint32_t*)(patch.ptr+patch.len-8);
    uint32_t crc_patch_e = *(uint32_t*)(patch.ptr+patch.len-4);

    uint32_t crc_in_a = crc32(in.ptr, in.len);
    uint32_t crc_patch_a = crc32(patch.ptr, patch.len-4);

    if (crc_patch_a != crc_patch_e) return bps_broken;

    size_t inlen, outlen, metadatalen;
    if (!decodenum(patchat, inlen)) return bps_too_big;
    if (!decodenum(patchat, outlen)) return bps_too_big;

    if (inlen != in.len || crc_in_a != crc_in_e) {
        if (in.len == outlen && crc_in_a == crc_out_e) error = bps_to_output;
        else error = bps_not_this;
        if (!accept_wrong_input) goto exit;
    }

    out->len = outlen;
    out->ptr = (uint8_t*)malloc(outlen);
    if (!out->ptr) return bps_out_of_mem;

    if (!decodenum(patchat, metadatalen)) error = bps_too_big;
    if (metadata && metadatalen) {
        metadata->len = metadatalen;
        metadata->ptr = (uint8_t*)malloc(metadatalen+1);
        if (!metadata->ptr) error = bps_out_of_mem;
        memcpy(metadata->ptr, patchat, metadatalen);
        metadata->ptr[metadatalen] = '\0';
        patchat += metadatalen;
    } else {
        patchat += metadatalen;
    }

    // Predecode all jobs
    std::vector<PatchJob> jobs;
    const uint8_t *patchscan = patchat;
    size_t outpos = 0, inpos = 0, patchpos = patchscan - patch.ptr;
    uint8_t* outstart = out->ptr;
    const uint8_t* instart = in.ptr;

    while (patchscan < patchend) {
        size_t thisinstr;
        if (!decodenum(patchscan, thisinstr)) error = bps_too_big;
        size_t length = (thisinstr >> 2) + 1;
        int action = thisinstr & 3;
        PatchJob job = {outpos, inpos, patchpos, length, action};
        if (action == SourceCopy || action == TargetCopy) {
            size_t param;
            if (!decodenum(patchscan, param)) error = bps_too_big;
            job.in_offset = param; // reuse for copy param
            job.patch_offset = patchscan - patch.ptr;
        } else if (action == TargetRead) {
            job.patch_offset = patchscan - patch.ptr;
            patchscan += length;
        }
        jobs.push_back(job);
        outpos += length;
        if (action == SourceRead) inpos += length;
        if (action == TargetRead) patchpos = job.patch_offset + length;
        else patchpos = patchscan - patch.ptr;
    }

    // Partition jobs into threaded and single-threaded
    std::vector<PatchJob> threaded, single;
    batch_jobs(jobs, threaded, single);

    // Launch threads for big jobs
    std::vector<std::thread> threads;
    for (const PatchJob& job : threaded) {
        threads.emplace_back([&, job](){
            if (job.action == SourceRead) {
                fast_copy(outstart + job.out_offset, instart + job.in_offset, job.length);
            } else if (job.action == TargetRead) {
                fast_copy(outstart + job.out_offset, patch.ptr + job.patch_offset, job.length);
            }
        });
    }
    // Single-threaded jobs (small and/or overlap/copy)
    for (const PatchJob& job : single) {
        switch (job.action) {
            case SourceRead:
                fast_copy(outstart + job.out_offset, instart + job.in_offset, job.length);
                break;
            case TargetRead:
                fast_copy(outstart + job.out_offset, patch.ptr + job.patch_offset, job.length);
                break;
            case SourceCopy: {
                size_t distance = job.in_offset >> 1;
                static const uint8_t* inreadat = instart;
                if ((job.in_offset & 1) == 0) inreadat += distance;
                else inreadat -= distance;
                fast_copy(outstart + job.out_offset, inreadat, job.length);
                inreadat += job.length;
                break;
            }
            case TargetCopy: {
                size_t distance = job.in_offset >> 1;
                static uint8_t* outreadat = outstart;
                if ((job.in_offset & 1) == 0) outreadat += distance;
                else outreadat -= distance;
                fast_copy(outstart + job.out_offset, outreadat, job.length);
                outreadat += job.length;
                break;
            }
        }
    }
    for (auto& t : threads) t.join();

    uint32_t crc_out_a = crc32(out->ptr, out->len);
    if (crc_out_a != crc_out_e) {
        error = bps_not_this;
        if (!accept_wrong_input) goto exit;
    }
    return error;

exit:
    free(out->ptr);
    out->len = 0;
    out->ptr = nullptr;
    if (metadata) {
        free(metadata->ptr);
        metadata->len = 0;
        metadata->ptr = nullptr;
    }
    return error;
}


#define write(val) \
			do { \
				out[outlen++]=(val); \
				if (outlen==outbuflen) \
				{ \
					outbuflen*=2; \
					uint8_t* newout=(uint8_t*)realloc(out, outbuflen); \
					if (!newout) { free(out); return bps_out_of_mem; } \
					out=newout; \
				} \
			} while(0)
#define write32(val) \
			do { \
				uint32_t tmp=(val); \
				write(tmp); \
				write(tmp>>8); \
				write(tmp>>16); \
				write(tmp>>24); \
			} while(0)
#define writenum(val) \
			do { \
				size_t tmpval=(val); \
				while (true) \
				{ \
					uint8_t tmpbyte=(tmpval&0x7F); \
					tmpval>>=7; \
					if (!tmpval) \
					{ \
						write(tmpbyte|0x80); \
						break; \
					} \
					write(tmpbyte); \
					tmpval--; \
				} \
			} while(0)

enum bpserror bps_create_linear(struct mem sourcemem, struct mem targetmem, struct mem metadata, struct mem * patchmem)
{
	if (sourcemem.len>=(SIZE_MAX>>2) - 16) return bps_too_big;//the 16 is just to be on the safe side, I don't think it's needed.
	if (targetmem.len>=(SIZE_MAX>>2) - 16) return bps_too_big;
	
	const uint8_t * source=sourcemem.ptr;
	const uint8_t * sourceend=sourcemem.ptr+sourcemem.len;
	if (sourcemem.len>targetmem.len) sourceend=sourcemem.ptr+targetmem.len;
	const uint8_t * targetbegin=targetmem.ptr;
	const uint8_t * target=targetmem.ptr;
	const uint8_t * targetend=targetmem.ptr+targetmem.len;
	
	const uint8_t * targetcopypos=targetbegin;
	
	size_t outbuflen=4096;
	uint8_t * out=(uint8_t*)malloc(outbuflen);
	if (!out) return bps_out_of_mem;
	size_t outlen=0;
	write('B');
	write('P');
	write('S');
	write('1');
	writenum(sourcemem.len);
	writenum(targetmem.len);
	writenum(metadata.len);
	for (size_t i=0;i<metadata.len;i++) write(metadata.ptr[i]);
	
	size_t mainContentPos=outlen;
	
	const uint8_t * lastknownchange=targetbegin;
	while (target<targetend)
	{
		size_t numunchanged=0;
		while (source+numunchanged<sourceend && source[numunchanged]==target[numunchanged]) numunchanged++;
		if (numunchanged>1 || numunchanged == (uintptr_t)(targetend-target))
		{
			//assert_shift((numunchanged-1), 2);
			writenum((numunchanged-1)<<2 | 0);//SourceRead
			source+=numunchanged;
			target+=numunchanged;
		}
		
		size_t numchanged=0;
		if (lastknownchange>target) numchanged=lastknownchange-target;
		while ((source+numchanged>=sourceend ||
		        source[numchanged]!=target[numchanged] ||
		        source[numchanged+1]!=target[numchanged+1] ||
		        source[numchanged+2]!=target[numchanged+2]) &&
		       target+numchanged<targetend)
		{
			numchanged++;
			if (source+numchanged>=sourceend) numchanged=targetend-target;
		}
		lastknownchange=target+numchanged;
		if (numchanged)
		{
			//assert_shift((numchanged-1), 2);
			size_t rle1start=(target==targetbegin);
			while (true)
			{
				if (
					target[rle1start-1]==target[rle1start+0] &&
					target[rle1start+0]==target[rle1start+1] &&
					target[rle1start+1]==target[rle1start+2] &&
					target[rle1start+2]==target[rle1start+3])
				{
					numchanged=rle1start;
					break;
				}
				if (
					target[rle1start-2]==target[rle1start+0] &&
					target[rle1start-1]==target[rle1start+1] &&
					target[rle1start+0]==target[rle1start+2] &&
					target[rle1start+1]==target[rle1start+3] &&
					target[rle1start+2]==target[rle1start+4])
				{
					numchanged=rle1start;
					break;
				}
				if (rle1start+3>=numchanged) break;
				rle1start++;
			}
			if (numchanged)
			{
				writenum((numchanged-1)<<2 | TargetRead);
				for (size_t i=0;i<numchanged;i++)
				{
					write(target[i]);
				}
				source+=numchanged;
				target+=numchanged;
			}
			if (target[-2]==target[0] && target[-1]==target[1] && target[0]==target[2])
			{
				//two-byte RLE
				size_t rlelen=0;
				while (target+rlelen<targetend && target[0]==target[rlelen+0] && target[1]==target[rlelen+1]) rlelen+=2;
				writenum((rlelen-1)<<2 | TargetCopy);
				writenum((target-targetcopypos-2)<<1);
				source+=rlelen;
				target+=rlelen;
				targetcopypos=target-2;
			}
			else if (target[-1]==target[0] && target[0]==target[1])
			{
				//one-byte RLE
				size_t rlelen=0;
				while (target+rlelen<targetend && target[0]==target[rlelen]) rlelen++;
				writenum((rlelen-1)<<2 | TargetCopy);
				writenum((target-targetcopypos-1)<<1);
				source+=rlelen;
				target+=rlelen;
				targetcopypos=target-1;
			}
		}
	}
	
	write32(crc32(sourcemem.ptr, sourcemem.len));
	write32(crc32(targetmem.ptr, targetmem.len));
	write32(crc32(out, outlen));
	
	patchmem->ptr=out;
	patchmem->len=outlen;
	
	//while this may look like it can be fooled by a patch containing exactly one of any other command, it
	//  can't, because the ones that aren't SourceRead requires an argument
	size_t i;
	for (i=mainContentPos;(out[i]&0x80)==0x00;i++) {}
	if (i==outlen-12-1) return bps_identical;
	
	return bps_ok;
}

#undef write_nocrc
#undef write
#undef writenum

void bps_free(struct mem mem)
{
	free(mem.ptr);
}
#undef error



struct bpsinfo bps_get_info(file* patch, bool changefrac)
{
#define error(why) do { ret.error=why; return ret; } while(0)
	struct bpsinfo ret;
	size_t len = patch->len();
	if (len<4+3+12) error(bps_broken);
	
	uint8_t top[256];
	if (!patch->read(top, 0, len>256 ? 256 : len)) error(bps_io);
	if (memcmp(top, "BPS1", 4)) error(bps_broken);
	
	const uint8_t* patchdat=top+4;
	if (!decodenum(patchdat, ret.size_in)) error(bps_too_big);
	if (!decodenum(patchdat, ret.size_out)) error(bps_too_big);
	
	if (!decodenum(patchdat, ret.meta_size)) error(bps_too_big);
	ret.meta_start = patchdat - top;
	
	uint8_t checksums[12];
	if (!patch->read(checksums, len-12, 12)) error(bps_io);
	ret.crc_in  = read32(checksums+0);
	ret.crc_out = read32(checksums+4);
	ret.crc_patch=read32(checksums+8);
	
	if (changefrac && ret.size_in>0)
	{
		//algorithm: each command adds its length to the numerator, unless it's above 32, in which case
		// it adds 32; or if it's SourceRead, in which case it adds 0
		//denominator is just input length
		uint8_t* patchbin=(uint8_t*)malloc(len);
		patch->read(patchbin, 0, len);
		size_t outpos=0; // position in the output file
		size_t changeamt=0; // change score
		const uint8_t* patchat=patchbin+(patchdat-top);
		
		size_t metasize;
		if (!decodenum(patchat, metasize)) error(bps_too_big);
		patchat+=metasize;
		
		const uint8_t* patchend=patchbin+len-12;
		
		while (patchat<patchend && outpos<ret.size_in)
		{
			size_t thisinstr;
			decodenum(patchat, thisinstr);
			size_t length=(thisinstr>>2)+1;
			int action=(thisinstr&3);
			int min_len_32 = (length<32 ? length : 32);
			
			switch (action)
			{
				case SourceRead:
				{
					changeamt+=0;
				}
				break;
				case TargetRead:
				{
					changeamt+=min_len_32;
					patchat+=length;
				}
				break;
				case SourceCopy:
				case TargetCopy:
				{
					changeamt+=min_len_32;
					size_t ignore;
					decodenum(patchat, ignore);
				}
				break;
			}
			outpos+=length;
		}
		if (patchat>patchend || outpos>ret.size_out) error(bps_broken);
		ret.change_num = (changeamt<ret.size_in ? changeamt : ret.size_in);
		ret.change_denom = ret.size_in;
		
		free(patchbin);
	}
	else
	{
		//this also happens if change fraction is not requested, but it's undefined behaviour anyways.
		ret.change_num=1;
		ret.change_denom=1;
	}
	
	ret.error=bps_ok;
	return ret;
}

#include <stdio.h>

#ifdef _WIN32
# ifdef _WIN64
#  ifdef __GNUC__
#   pragma GCC diagnostic ignored "-Wformat"
#  endif
#  define z "I"
# else
#  define z ""
# endif
#else
# define z "z"
#endif

void bps_disassemble(struct mem patch, FILE* out)
{
#define read8() (*(patchat++))
#define decodeto(var) decodenum(patchat, var)
	const uint8_t * patchat=patch.ptr;
	const uint8_t * patchend=patch.ptr+patch.len-12;
	
	if (read8() != 'B' || read8() != 'P' || read8() != 'S' || read8() != '1')
	{
		fprintf(out, "Not a BPS patch\n");
		return;
	}
	
	size_t inreadat = 0;
	size_t inlen;
	decodeto(inlen);
	
	size_t outat = 0;
	size_t outreadat = 0;
	size_t outlen;
	decodeto(outlen);
	
	size_t metadatalen;
	decodeto(metadatalen);
	
	patchat += metadatalen;
	
	while (patchat<patchend)
	{
		size_t thisinstr;
		decodeto(thisinstr);
		size_t length=(thisinstr>>2)+1;
		int action=(thisinstr&3);
		
		switch (action)
		{
			case SourceRead:
			{
				fprintf(out, "SourceRead %" z "u from %" z "u to %" z "u\n", length, outat, outat);
				outat += length;
			}
			break;
			case TargetRead:
			{
				fprintf(out, "TargetRead %" z "u to %" z "u\n", length, outat);
				patchat += length;
				outat += length;
			}
			break;
			case SourceCopy:
			{
				size_t encodeddistance;
				decodeto(encodeddistance);
				size_t distance=encodeddistance>>1;
				if ((encodeddistance&1)==0) inreadat+=distance;
				else inreadat-=distance;
				
				fprintf(out, "SourceCopy %" z "u from %" z "u to %" z "u (rel %+" z "i)\n", length, inreadat, outat, inreadat-outat);
				inreadat += length;
				outat += length;
			}
			break;
			case TargetCopy:
			{
				size_t encodeddistance;
				decodeto(encodeddistance);
				size_t distance=encodeddistance>>1;
				if ((encodeddistance&1)==0) outreadat+=distance;
				else outreadat-=distance;
				
				fprintf(out, "TargetCopy %" z "u from %" z "u to %" z "u (rel %+" z "i)\n", length, outreadat, outat, outreadat-outat);
				outreadat += length;
				outat += length;
			}
			break;
		}
	}
	
	if (patchat != patchend)
		fprintf(out, "WARNING: patch pointer at %" z "u != %" z "u, corrupt patch?\n", patchat-patch.ptr, patchend-patch.ptr);
	if (outat != outlen)
		fprintf(out, "WARNING: output pointer at %" z "u != %" z "u, corrupt patch?\n", outat, outlen);
#undef read8
#undef decodeto
}


#ifdef BPS_DEBUG
#warning Disable this in release versions.

//Congratulations, you found the undocumented feature! It compares two equivalent BPS patches and tells where each one is more compact.
//It will crash or otherwise misbehave on invalid or non-equivalent patches.
//Have fun.
void bps_compare(struct mem patch1mem, struct mem patch2mem)
{
	const uint8_t * patch[2]={patch1mem.ptr, patch2mem.ptr};
	size_t patchpos[2]={0,0};
	size_t patchlen[2]={patch1mem.len-12, patch2mem.len-12};
	size_t patchoutpos[2]={0,0};
	
	size_t patchcopypos[2][4]={0,0};//[0] and [1] are unused, but this is just debug code, it doesn't need to be neat.
	
#define read8(id) (patch[id][patchpos[id]++])
#define decodeto(id, var) \
				do { \
					var=0; \
					int shift=0; \
					while (true) \
					{ \
						uint8_t next=read8(id); \
						size_t addthis=(next&0x7F)<<shift; \
						var+=addthis; \
						if (next&0x80) break; \
						shift+=7; \
						var+=1<<shift; \
					} \
				} while(false)
	
	size_t lastmatch=0;
	size_t patchposatmatch[2]={0,0};
	
	size_t outlen;
	patch[0]+=4; patch[1]+=4;//BPS1
	size_t tempuint;
	decodeto(0, tempuint); decodeto(1, tempuint);//source-size
	decodeto(0, outlen); decodeto(1, outlen);//target-size
	decodeto(0, tempuint); patch[0]+=tempuint;//metadata
	decodeto(1, tempuint); patch[1]+=tempuint;//metadata
	
	bool show=false;
	while (patchpos[0]<patchlen[0] && patchpos[1]<patchlen[1])
	{
		bool step[2]={(patchoutpos[0]<=patchoutpos[1]), (patchoutpos[0]>=patchoutpos[1])};
		char describe[2][256];
		for (int i=0;i<2;i++)
		{
			if (step[i])
			{
				size_t patchposstart=patchpos[i];
				decodeto(i, tempuint);
				size_t len=(tempuint>>2)+1;
				patchoutpos[i]+=len;
				int action=(tempuint&3);
//enum { SourceRead, TargetRead, SourceCopy, TargetCopy };
				const char * actionnames[]={"SourceRead", "TargetRead", "SourceCopy", "TargetCopy"};
				if (action==TargetRead) patchpos[i]+=len;
				if (action==SourceCopy || action==TargetCopy)
				{
					decodeto(i, tempuint);
					int delta = tempuint>>1;
					if (tempuint&1) delta=-delta;
					patchcopypos[i][action]+=delta;
					sprintf(describe[i], "%s from %zu (%+i) for %zu in %zu",  actionnames[action], patchcopypos[i][action], delta, len, patchpos[i]-patchposstart);
					patchcopypos[i][action]+=len;
				}
				else sprintf(describe[i], "%s from %zu for %zu in %zu",  actionnames[action], patchoutpos[i], len, patchpos[i]-patchposstart);
				if (!step[i^1])
				{
					printf("%i: %s\n", i+1, describe[i]);
					show=true;
				}
			}
		}
		if (step[0] && step[1])
		{
			if (!strcmp(describe[0], describe[1])) /*printf("3: %s\n", describe[0])*/;
			else
			{
				printf("1: %s\n2: %s\n", describe[0], describe[1]);
				show=true;
			}
		}
		if (patchoutpos[0]==patchoutpos[1])
		{
			size_t used[2]={patchpos[0]-patchposatmatch[0], patchpos[1]-patchposatmatch[1]};
			char which='=';
			if (used[0]<used[1]) which='+';
			if (used[0]>used[1]) which='-';
			if (show)
			{
				printf("%c: %zu,%zu bytes since last match (%zu)\n", which, used[0], used[1], patchoutpos[0]);
				show=false;
			}
			patchposatmatch[0]=patchpos[0];
			patchposatmatch[1]=patchpos[1];
			lastmatch=patchoutpos[0];
		}
	}
#undef read8
#undef decodeto
}

static struct mem ReadWholeFile(const char * filename)
{
	struct mem null = {NULL, 0};
	
	FILE * file=fopen(filename, "rb");
	if (!file) return null;
	fseek(file, 0, SEEK_END);
	size_t len=ftell(file);
	fseek(file, 0, SEEK_SET);
	unsigned char * data=(unsigned char*)malloc(len);
	size_t truelen=fread(data, 1,len, file);
	fclose(file);
	if (len!=truelen)
	{
		free(data);
		return null;
	}
	
	struct mem ret = { (unsigned char*)data, len };
	return ret;
}

int main(int argc,char**argv)
{
	if (argc==1) puts("bad arguments");
	if (argc==2) bps_dump(ReadWholeFile(argv[1]));
	if (argc==3) bps_compare(ReadWholeFile(argv[1]),ReadWholeFile(argv[2]));
}
#endif
