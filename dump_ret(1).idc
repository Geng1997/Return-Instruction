// dump bytes from ``ret'' of a function to the next 32 bytes
#include <idc.idc>

static main() {
  auto ea,x;
  auto end;
  auto i, b, j, addr, head, label;
  auto f = fopen(ARGV[1], "ab+");
  auto id = atol(ARGV[2]);
  auto MAX_LEN = 16;
  
  SetShortPrm(INF_AF2, GetShortPrm(INF_AF2) | AF2_DODATA);
  Message("Waiting for the end of the auto analysis...\n");
  Wait();
  for ( ea=NextFunction(0); ea != BADADDR; ea=NextFunction(ea) ) {
	end = FindFuncEnd(ea);
	for (i=ea;i<end;i++) {
		b = Byte(i);
		if (b == 0xC3 || b == 0xC2 || b == 0xCB || b == 0xCA) { // ret?
			head = ItemHead(i);

			// prev 16 bytes
			addr = PrevAddr(i);
			for (j=0;j<MAX_LEN;j++) {
				if (addr != BADADDR) {
					fputc(Byte(addr), f);
				} else {
					fputc(0xff, f);
				}
				addr = PrevAddr(addr);
			}

            // next 16 bytes
            addr = i;
			for (j=0;j<MAX_LEN;j++) {
				if (addr != BADADDR) {
					fputc(Byte(addr), f);
				} else {
					fputc(0xff, f);
				}
				addr = NextAddr(addr);
			}

			// label
			// 0 not an instruction, 1 ret, 2 ret and tail
			label = 0;
			if (head == i) { // ret instruction?
				label = 1;
				if (end == ItemEnd(i)) { // function last instruction?
				    label = 2;
				}
			}
			fputc(label, f);
			writelong(f, i, 0); // location
			writelong(f, id, 0); // file id
		}
	}
  }
  
  fclose(f);
  Message("Dump done.\n");
  Exit(0); 
}
