
;; Function main (main, funcdef_no=0, decl_uid=2826, symbol_order=49) (executed once)

;; 12 loops found
;;
;; Loop 0
;;  header 0, latch 1
;;  depth 0, outer -1
;;  nodes: 0 1 2 3 4 47 45 5 6 7 46 8 9 43 10 11 12 44 13 14 15 16 42 17 18 19 20 41 21 38 22 23 24 25 40 26 27 39 28 29 30 31 37 32 33 34 35 36
;;
;; Loop 2
;;  header 30, latch 37
;;  depth 1, outer 0
;;  nodes: 30 37 31
;;
;; Loop 3
;;  header 22, latch 38
;;  depth 1, outer 0
;;  nodes: 22 38 28 27 26 25 24 40 23 39
;;
;; Loop 4
;;  header 23, latch 39
;;  depth 2, outer 3
;;  nodes: 23 39 27 26 25 24 40
;;
;; Loop 5
;;  header 24, latch 40
;;  depth 3, outer 4
;;  nodes: 24 40 25
;;
;; Loop 6
;;  header 21, latch 41
;;  depth 1, outer 0
;;  nodes: 21 41 20 19 18 17 42
;;
;; Loop 7
;;  header 17, latch 42
;;  depth 2, outer 6
;;  nodes: 17 42 18
;;
;; Loop 8
;;  header 10, latch 43
;;  depth 1, outer 0
;;  nodes: 10 43 16 15 13 14 12 11 44
;;
;; Loop 9
;;  header 11, latch 44
;;  depth 2, outer 8
;;  nodes: 11 44 12
;;
;; Loop 10
;;  header 5, latch 45
;;  depth 1, outer 0
;;  nodes: 5 45 9 8 7 6 46
;;
;; Loop 11
;;  header 6, latch 46
;;  depth 2, outer 10
;;  nodes: 6 46 7
;;
;; Loop 1
;;  header 3, latch 47
;;  depth 1, outer 0
;;  nodes: 3 47 4
;; 2 succs { 3 }
;; 3 succs { 4 }
;; 4 succs { 47 36 }
;; 47 succs { 3 }
;; 45 succs { 5 }
;; 5 succs { 6 }
;; 6 succs { 7 }
;; 7 succs { 46 8 }
;; 46 succs { 6 }
;; 8 succs { 9 }
;; 9 succs { 45 35 }
;; 43 succs { 10 }
;; 10 succs { 11 }
;; 11 succs { 12 }
;; 12 succs { 44 13 }
;; 44 succs { 11 }
;; 13 succs { 15 14 }
;; 14 succs { 15 }
;; 15 succs { 16 }
;; 16 succs { 43 34 }
;; 42 succs { 17 }
;; 17 succs { 18 }
;; 18 succs { 42 19 }
;; 19 succs { 20 }
;; 20 succs { 41 33 }
;; 41 succs { 21 }
;; 21 succs { 17 }
;; 38 succs { 22 }
;; 22 succs { 23 }
;; 23 succs { 24 }
;; 24 succs { 25 }
;; 25 succs { 40 26 }
;; 40 succs { 24 }
;; 26 succs { 27 }
;; 27 succs { 39 28 }
;; 39 succs { 23 }
;; 28 succs { 38 29 }
;; 29 succs { 30 }
;; 30 succs { 31 }
;; 31 succs { 37 32 }
;; 37 succs { 30 }
;; 32 succs { 1 }
;; 33 succs { 22 }
;; 34 succs { 21 }
;; 35 succs { 10 }
;; 36 succs { 5 }
Merging blocks 3 and 4
Merging blocks 6 and 7
Merging blocks 8 and 9
Merging blocks 11 and 12
Merging blocks 15 and 16
Merging blocks 17 and 18
Merging blocks 19 and 20
Merging blocks 24 and 25
Merging blocks 26 and 27
Merging blocks 30 and 31
Removing basic block 37
Removing basic block 38
Removing basic block 39
Removing basic block 40
Removing basic block 41
Removing basic block 42
Removing basic block 43
Removing basic block 44
Removing basic block 45
Removing basic block 46
Removing basic block 47
main ()
{
  int i;
  int j;
  int j1;
  int j2;
  double * stddev;
  double * mean;
  double * symmat;
  double * data;
  int i;
  long unsigned int _15;
  long unsigned int _16;
  double * _17;
  int _19;
  double _20;
  double _21;
  long unsigned int _25;
  long unsigned int _26;
  double * _27;
  double _28;
  long unsigned int _44;
  long unsigned int _45;
  double * _46;
  double _47;
  int _49;
  int _50;
  long unsigned int _51;
  long unsigned int _52;
  double * _53;
  double _54;
  double _55;
  double _57;
  double _58;
  long unsigned int _61;
  long unsigned int _62;
  double * _63;
  double _64;
  int _66;
  int _67;
  long unsigned int _68;
  long unsigned int _69;
  double * _70;
  double _71;
  double * _72;
  double _73;
  double _74;
  double _75;
  double _76;
  double _78;
  double _79;
  double _80;
  double _81;
  int _84;
  int _86;
  long unsigned int _87;
  long unsigned int _88;
  double * _89;
  double _90;
  long unsigned int _91;
  long unsigned int _92;
  double * _93;
  double _94;
  double _95;
  double * _96;
  double _97;
  double _98;
  double _99;
  int _103;
  long unsigned int _104;
  long unsigned int _105;
  double * _106;
  int _108;
  int _110;
  long unsigned int _111;
  long unsigned int _112;
  double * _113;
  double _114;
  int _116;
  int _117;
  long unsigned int _118;
  long unsigned int _119;
  double * _120;
  double _121;
  int _122;
  long unsigned int _123;
  long unsigned int _124;
  double * _125;
  double _126;
  double _127;
  double _128;
  int _130;
  int _131;
  long unsigned int _132;
  long unsigned int _133;
  double * _134;
  double _135;

  <bb 2>:
  [corr_ref.c : 86:51] data_7 = malloc (524288);
  [corr_ref.c : 87:53] symmat_9 = malloc (524288);
  [corr_ref.c : 88:49] mean_11 = malloc (2048);
  [corr_ref.c : 89:51] stddev_13 = malloc (2048);
  [corr_ref.c : 91:13] srand (5497);

  <bb 3>:
  # i_165 = PHI <[corr_ref.c : 92:5] i_23(3), [corr_ref.c : 92:15] 0(2)>
  [corr_ref.c : 93:15] _15 = (long unsigned int) i_165;
  [corr_ref.c : 93:15] _16 = _15 * 8;
  [corr_ref.c : 93:15] _17 = data_7 + _16;
  [corr_ref.c : 93:31] _19 = rand ();
  [corr_ref.c : 93:33] _20 = (double) _19;
  [corr_ref.c : 93:33] _21 = _20 / 2.147483647e+9;
  [corr_ref.c : 93:50] [corr_ref.c : 93] *_17 = _21;
  [corr_ref.c : 92:5] i_23 = i_165 + 1;
  [corr_ref.c : 92:5] if (i_23 != 65536)
    goto <bb 3>;
  else
    goto <bb 26>;

  <bb 4>:
  # j_197 = PHI <[corr_ref.c : 24:4] j_59(6), j_189(26)>
  [corr_ref.c : 26:11] _44 = (long unsigned int) j_197;
  [corr_ref.c : 26:11] _45 = _44 * 8;
  [corr_ref.c : 26:11] _46 = mean_11 + _45;
  [corr_ref.c : 26:19] *_46 = 0.0;

  <bb 5>:
  # i_198 = PHI <[corr_ref.c : 28:6] i_56(5), [corr_ref.c : 28:16] 0(4)>
  [corr_ref.c : 30:26] _47 = *_46;
  [corr_ref.c : 30:21] _49 = i_198 * 256;
  [corr_ref.c : 30:23] _50 = j_197 + _49;
  [corr_ref.c : 30:25] _51 = (long unsigned int) _50;
  [corr_ref.c : 30:25] _52 = _51 * 8;
  [corr_ref.c : 30:25] _53 = data_7 + _52;
  [corr_ref.c : 30:25] _54 = *_53;
  [corr_ref.c : 30:26] _55 = _47 + _54;
  [corr_ref.c : 30:26] *_46 = _55;
  [corr_ref.c : 28:6] i_56 = i_198 + 1;
  [corr_ref.c : 28:6] if (i_56 != 256)
    goto <bb 5>;
  else
    goto <bb 6>;

  <bb 6>:
  [corr_ref.c : 33:28] _57 = *_46;
  [corr_ref.c : 33:28] _58 = _57 / 3.214212e+6;
  [corr_ref.c : 33:28] *_46 = _58;
  [corr_ref.c : 24:4] j_59 = j_197 + 1;
  [corr_ref.c : 24:4] if (j_59 != 256)
    goto <bb 4>;
  else
    goto <bb 25>;

  <bb 7>:
  # j_199 = PHI <[corr_ref.c : 37:4] j_82(11), j_177(25)>
  [corr_ref.c : 39:14] _61 = (long unsigned int) j_199;
  [corr_ref.c : 39:14] _62 = _61 * 8;
  [corr_ref.c : 39:14] _63 = stddev_13 + _62;
  [corr_ref.c : 39:22] *_63 = 0.0;

  <bb 8>:
  # i_200 = PHI <[corr_ref.c : 41:3] i_77(8), [corr_ref.c : 41:13] 0(7)>
  [corr_ref.c : 43:66] _64 = *_63;
  [corr_ref.c : 43:50] _66 = i_200 * 256;
  [corr_ref.c : 43:52] _67 = j_199 + _66;
  [corr_ref.c : 43:54] _68 = (long unsigned int) _67;
  [corr_ref.c : 43:54] _69 = _68 * 8;
  [corr_ref.c : 43:54] _70 = data_7 + _69;
  [corr_ref.c : 43:54] _71 = *_70;
  [corr_ref.c : 43:64] _72 = mean_11 + _62;
  [corr_ref.c : 43:64] _73 = *_72;
  [corr_ref.c : 43:56] _74 = _71 - _73;
  [corr_ref.c : 43:41] _75 = _74 * _74;
  [corr_ref.c : 43:66] _76 = _64 + _75;
  [corr_ref.c : 43:66] *_63 = _76;
  [corr_ref.c : 41:3] i_77 = i_200 + 1;
  [corr_ref.c : 41:3] if (i_77 != 256)
    goto <bb 8>;
  else
    goto <bb 9>;

  <bb 9>:
  [corr_ref.c : 46:23] _78 = *_63;
  [corr_ref.c : 46:23] _79 = _78 / 3.214212e+6;
  [corr_ref.c : 46:23] *_63 = _79;
  [corr_ref.c : 47:15] _80 = sqrt (_79);
  [corr_ref.c : 47:44] *_63 = _80;
  [corr_ref.c : 48:50] if (_80 <= 4.999999888241291046142578125e-3)
    goto <bb 11>;
  else
    goto <bb 10>;

  <bb 10>:

  <bb 11>:
  # _81 = PHI <[corr_ref.c : 48:50] 1.0e+0(9), [corr_ref.c : 48:50] _80(10)>
  [corr_ref.c : 48:50] *_63 = _81;
  [corr_ref.c : 37:4] j_82 = j_199 + 1;
  [corr_ref.c : 37:4] if (j_82 != 256)
    goto <bb 7>;
  else
    goto <bb 24>;

  <bb 12>:
  # j_202 = PHI <[corr_ref.c : 54:3] j_100(12), 0(14)>
  [corr_ref.c : 56:26] _84 = i_201 * 256;
  [corr_ref.c : 56:26] _86 = _84 + j_202;
  [corr_ref.c : 56:26] _87 = (long unsigned int) _86;
  [corr_ref.c : 56:26] _88 = _87 * 8;
  [corr_ref.c : 56:26] _89 = data_7 + _88;
  [corr_ref.c : 56:26] _90 = *_89;
  [corr_ref.c : 56:25] _91 = (long unsigned int) j_202;
  [corr_ref.c : 56:25] _92 = _91 * 8;
  [corr_ref.c : 56:25] _93 = mean_11 + _92;
  [corr_ref.c : 56:25] _94 = *_93;
  [corr_ref.c : 56:26] _95 = _90 - _94;
  [corr_ref.c : 56:26] *_89 = _95;
  [corr_ref.c : 57:42] _96 = stddev_13 + _92;
  [corr_ref.c : 57:42] _97 = *_96;
  [corr_ref.c : 57:33] _98 = _97 * 1.792822355951642975924187339842319488525390625e+3;
  [corr_ref.c : 57:45] _99 = _95 / _98;
  [corr_ref.c : 57:45] *_89 = _99;
  [corr_ref.c : 54:3] j_100 = j_202 + 1;
  [corr_ref.c : 54:3] if (j_100 != 256)
    goto <bb 12>;
  else
    goto <bb 13>;

  <bb 13>:
  [corr_ref.c : 52:4] i_101 = i_201 + 1;
  [corr_ref.c : 52:4] if (i_101 != 256)
    goto <bb 14>;
  else
    goto <bb 23>;

  <bb 14>:
  # i_201 = PHI <[corr_ref.c : 52:4] i_101(13), i_181(24)>
  goto <bb 12>;

  <bb 15>:
  # j1_203 = PHI <[corr_ref.c : 62:4] j1_102(19), j1_182(23)>
  [corr_ref.c : 64:14] _103 = j1_203 * 257;
  [corr_ref.c : 64:17] _104 = (long unsigned int) _103;
  [corr_ref.c : 64:17] _105 = _104 * 8;
  [corr_ref.c : 64:17] _106 = symmat_9 + _105;
  [corr_ref.c : 64:25] *_106 = 1.0e+0;
  [corr_ref.c : 66:17] j1_107 = j1_203 + 1;

  <bb 16>:
  # j1_204 = PHI <[corr_ref.c : 66:3] j1_136(18), [corr_ref.c : 66:17] j1_107(15)>
  [corr_ref.c : 68:15] _108 = j1_203 * 256;
  [corr_ref.c : 68:17] _110 = _108 + j1_204;
  [corr_ref.c : 68:20] _111 = (long unsigned int) _110;
  [corr_ref.c : 68:20] _112 = _111 * 8;
  [corr_ref.c : 68:20] _113 = symmat_9 + _112;
  [corr_ref.c : 68:28] *_113 = 0.0;

  <bb 17>:
  # i_205 = PHI <[corr_ref.c : 70:6] i_129(17), [corr_ref.c : 70:16] 0(16)>
  [corr_ref.c : 72:56] _114 = *_113;
  [corr_ref.c : 72:34] _116 = i_205 * 256;
  [corr_ref.c : 72:36] _117 = j1_203 + _116;
  [corr_ref.c : 72:39] _118 = (long unsigned int) _117;
  [corr_ref.c : 72:39] _119 = _118 * 8;
  [corr_ref.c : 72:39] _120 = data_7 + _119;
  [corr_ref.c : 72:39] _121 = *_120;
  [corr_ref.c : 72:51] _122 = j1_204 + _116;
  [corr_ref.c : 72:54] _123 = (long unsigned int) _122;
  [corr_ref.c : 72:54] _124 = _123 * 8;
  [corr_ref.c : 72:54] _125 = data_7 + _124;
  [corr_ref.c : 72:54] _126 = *_125;
  [corr_ref.c : 72:41] _127 = _121 * _126;
  [corr_ref.c : 72:56] _128 = _114 + _127;
  [corr_ref.c : 72:56] *_113 = _128;
  [corr_ref.c : 70:6] i_129 = i_205 + 1;
  [corr_ref.c : 70:6] if (i_129 != 256)
    goto <bb 17>;
  else
    goto <bb 18>;

  <bb 18>:
  [corr_ref.c : 75:15] _130 = j1_204 * 256;
  [corr_ref.c : 75:17] _131 = j1_203 + _130;
  [corr_ref.c : 75:20] _132 = (long unsigned int) _131;
  [corr_ref.c : 75:20] _133 = _132 * 8;
  [corr_ref.c : 75:20] _134 = symmat_9 + _133;
  [corr_ref.c : 75:38] _135 = *_113;
  [corr_ref.c : 75:39] *_134 = _135;
  [corr_ref.c : 66:3] j1_136 = j1_204 + 1;
  [corr_ref.c : 66:3] if (j1_136 <= 255)
    goto <bb 16>;
  else
    goto <bb 19>;

  <bb 19>:
  # j1_102 = PHI <[corr_ref.c : 62:4] j1_107(18)>
  [corr_ref.c : 62:4] if (j1_102 != 255)
    goto <bb 15>;
  else
    goto <bb 20>;

  <bb 20>:
  [corr_ref.c : 79:32] MEM[(double *)symmat_9 + 524280B] = 1.0e+0;

  <bb 21>:
  # i_193 = PHI <[corr_ref.c : 97:2] i_30(21), [corr_ref.c : 97:12] 0(20)>
  [corr_ref.c : 98:28] _25 = (long unsigned int) i_193;
  [corr_ref.c : 98:28] _26 = _25 * 8;
  [corr_ref.c : 98:28] _27 = symmat_9 + _26;
  [corr_ref.c : 98:29] _28 = [corr_ref.c : 98] *_27;
  [/usr/include/x86_64-linux-gnu/bits/stdio2.h : 104:72] __printf_chk (1, "%.15f,", _28);
  [corr_ref.c : 97:2] i_30 = i_193 + 1;
  [corr_ref.c : 97:2] if (i_30 != 65536)
    goto <bb 21>;
  else
    goto <bb 22>;

  <bb 22>:
  return 0;

  <bb 23>:
  # j1_182 = PHI <0(13)>
  goto <bb 15>;

  <bb 24>:
  # i_181 = PHI <0(11)>
  goto <bb 14>;

  <bb 25>:
  # j_177 = PHI <0(6)>
  goto <bb 7>;

  <bb 26>:
  # j_189 = PHI <0(3)>
  goto <bb 4>;

}


