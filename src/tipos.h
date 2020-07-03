#ifndef _tipos_h
#define _tipos_h

//Macro performing 4^n
#define POW(k) (1U << 2*(k))

//Typedef
typedef unsigned short ushort;
typedef long int lint;
typedef unsigned int uint;

//0 disabled; 1 enable
const int DBG = 0;

typedef struct Seq
{
  char *header;
  char *read;
  char *data;
  int len;
} Seq;

typedef struct read// Used to read sequences
{
  char *data;
  int *length;
  lint *start;
  struct Freq *freq;
} Read;

typedef struct Freq
{
  lint kmer;
  int count;
} Freq;

#endif
