
;; Function void correlation(double*, double*, double*, double*) (_Z11correlationPdS_S_S_, funcdef_no=48, decl_uid=4651, symbol_order=48)

void correlation(double*, double*, double*, double*) (double * data, double * mean, double * stddev, double * symmat)
{
  int j2;
  int j1;
  int j;
  int i;
  double * D.4761;
  double * D.4760;
  long unsigned int D.4759;
  long unsigned int D.4758;
  int D.4757;
  int D.4756;
  double D.4755;
  double D.4754;
  double D.4753;
  double * D.4752;
  long unsigned int D.4751;
  long unsigned int D.4750;
  int D.4749;
  double D.4748;
  double * D.4747;
  long unsigned int D.4746;
  long unsigned int D.4745;
  int D.4744;
  double D.4743;
  double * D.4742;
  long unsigned int D.4741;
  long unsigned int D.4740;
  int D.4739;
  int D.4738;
  double * D.4737;
  long unsigned int D.4736;
  long unsigned int D.4735;
  int D.4734;
  double D.4733;
  double D.4732;
  double iftmp.0;
  double D.4727;
  double D.4726;
  double D.4725;
  double D.4724;
  double D.4723;
  double D.4722;
  double * D.4721;
  double D.4720;
  double D.4719;
  double D.4718;
  double * D.4717;
  long unsigned int D.4716;
  long unsigned int D.4715;
  int D.4714;
  int D.4713;
  double D.4712;
  double * D.4711;
  long unsigned int D.4710;
  long unsigned int D.4709;

  [corr_ref.c : 24:14] j = 0;
  [corr_ref.c : 24:4] goto <D.4664>;
  <D.4663>:
  [corr_ref.c : 26:11] D.4709 = (long unsigned int) j;
  [corr_ref.c : 26:11] D.4710 = D.4709 * 8;
  [corr_ref.c : 26:11] D.4711 = mean + D.4710;
  [corr_ref.c : 26:19] [corr_ref.c : 26] *D.4711 = 0.0;
  [corr_ref.c : 28:16] i = 0;
  [corr_ref.c : 28:6] goto <D.4662>;
  <D.4661>:
  [corr_ref.c : 30:26] D.4709 = (long unsigned int) j;
  [corr_ref.c : 30:26] D.4710 = D.4709 * 8;
  [corr_ref.c : 30:26] D.4711 = mean + D.4710;
  [corr_ref.c : 30:26] D.4709 = (long unsigned int) j;
  [corr_ref.c : 30:26] D.4710 = D.4709 * 8;
  [corr_ref.c : 30:26] D.4711 = mean + D.4710;
  [corr_ref.c : 30:26] D.4712 = [corr_ref.c : 30] *D.4711;
  [corr_ref.c : 30:21] D.4713 = i * 256;
  [corr_ref.c : 30:23] D.4714 = D.4713 + j;
  [corr_ref.c : 30:25] D.4715 = (long unsigned int) D.4714;
  [corr_ref.c : 30:25] D.4716 = D.4715 * 8;
  [corr_ref.c : 30:25] D.4717 = data + D.4716;
  [corr_ref.c : 30:25] D.4718 = [corr_ref.c : 30] *D.4717;
  [corr_ref.c : 30:26] D.4719 = D.4712 + D.4718;
  [corr_ref.c : 30:26] [corr_ref.c : 30] *D.4711 = D.4719;
  [corr_ref.c : 28:6] i = i + 1;
  <D.4662>:
  [corr_ref.c : 28:6] if (i <= 255) goto <D.4661>; else goto <D.4659>;
  <D.4659>:
  [corr_ref.c : 33:28] D.4709 = (long unsigned int) j;
  [corr_ref.c : 33:28] D.4710 = D.4709 * 8;
  [corr_ref.c : 33:28] D.4711 = mean + D.4710;
  [corr_ref.c : 33:28] D.4709 = (long unsigned int) j;
  [corr_ref.c : 33:28] D.4710 = D.4709 * 8;
  [corr_ref.c : 33:28] D.4711 = mean + D.4710;
  [corr_ref.c : 33:28] D.4712 = [corr_ref.c : 33] *D.4711;
  [corr_ref.c : 33:28] D.4720 = D.4712 / 3.214212e+6;
  [corr_ref.c : 33:28] [corr_ref.c : 33] *D.4711 = D.4720;
  [corr_ref.c : 24:4] j = j + 1;
  <D.4664>:
  [corr_ref.c : 24:4] if (j <= 255) goto <D.4663>; else goto <D.4657>;
  <D.4657>:
  [corr_ref.c : 37:14] j = 0;
  [corr_ref.c : 37:4] goto <D.4672>;
  <D.4671>:
  [corr_ref.c : 39:14] D.4709 = (long unsigned int) j;
  [corr_ref.c : 39:14] D.4710 = D.4709 * 8;
  [corr_ref.c : 39:14] D.4721 = stddev + D.4710;
  [corr_ref.c : 39:22] [corr_ref.c : 39] *D.4721 = 0.0;
  [corr_ref.c : 41:13] i = 0;
  [corr_ref.c : 41:3] goto <D.4670>;
  <D.4669>:
  [corr_ref.c : 43:66] D.4709 = (long unsigned int) j;
  [corr_ref.c : 43:66] D.4710 = D.4709 * 8;
  [corr_ref.c : 43:66] D.4721 = stddev + D.4710;
  [corr_ref.c : 43:66] D.4709 = (long unsigned int) j;
  [corr_ref.c : 43:66] D.4710 = D.4709 * 8;
  [corr_ref.c : 43:66] D.4721 = stddev + D.4710;
  [corr_ref.c : 43:66] D.4722 = [corr_ref.c : 43] *D.4721;
  [corr_ref.c : 43:24] D.4713 = i * 256;
  [corr_ref.c : 43:26] D.4714 = D.4713 + j;
  [corr_ref.c : 43:28] D.4715 = (long unsigned int) D.4714;
  [corr_ref.c : 43:28] D.4716 = D.4715 * 8;
  [corr_ref.c : 43:28] D.4717 = data + D.4716;
  [corr_ref.c : 43:28] D.4718 = [corr_ref.c : 43] *D.4717;
  [corr_ref.c : 43:38] D.4709 = (long unsigned int) j;
  [corr_ref.c : 43:38] D.4710 = D.4709 * 8;
  [corr_ref.c : 43:38] D.4711 = mean + D.4710;
  [corr_ref.c : 43:38] D.4712 = [corr_ref.c : 43] *D.4711;
  [corr_ref.c : 43:30] D.4723 = D.4718 - D.4712;
  [corr_ref.c : 43:50] D.4713 = i * 256;
  [corr_ref.c : 43:52] D.4714 = D.4713 + j;
  [corr_ref.c : 43:54] D.4715 = (long unsigned int) D.4714;
  [corr_ref.c : 43:54] D.4716 = D.4715 * 8;
  [corr_ref.c : 43:54] D.4717 = data + D.4716;
  [corr_ref.c : 43:54] D.4718 = [corr_ref.c : 43] *D.4717;
  [corr_ref.c : 43:64] D.4709 = (long unsigned int) j;
  [corr_ref.c : 43:64] D.4710 = D.4709 * 8;
  [corr_ref.c : 43:64] D.4711 = mean + D.4710;
  [corr_ref.c : 43:64] D.4712 = [corr_ref.c : 43] *D.4711;
  [corr_ref.c : 43:56] D.4723 = D.4718 - D.4712;
  [corr_ref.c : 43:41] D.4724 = D.4723 * D.4723;
  [corr_ref.c : 43:66] D.4725 = D.4722 + D.4724;
  [corr_ref.c : 43:66] [corr_ref.c : 43] *D.4721 = D.4725;
  [corr_ref.c : 41:3] i = i + 1;
  <D.4670>:
  [corr_ref.c : 41:3] if (i <= 255) goto <D.4669>; else goto <D.4667>;
  <D.4667>:
  [corr_ref.c : 46:23] D.4709 = (long unsigned int) j;
  [corr_ref.c : 46:23] D.4710 = D.4709 * 8;
  [corr_ref.c : 46:23] D.4721 = stddev + D.4710;
  [corr_ref.c : 46:23] D.4709 = (long unsigned int) j;
  [corr_ref.c : 46:23] D.4710 = D.4709 * 8;
  [corr_ref.c : 46:23] D.4721 = stddev + D.4710;
  [corr_ref.c : 46:23] D.4722 = [corr_ref.c : 46] *D.4721;
  [corr_ref.c : 46:23] D.4726 = D.4722 / 3.214212e+6;
  [corr_ref.c : 46:23] [corr_ref.c : 46] *D.4721 = D.4726;
  [corr_ref.c : 47:11] D.4709 = (long unsigned int) j;
  [corr_ref.c : 47:11] D.4710 = D.4709 * 8;
  [corr_ref.c : 47:11] D.4721 = stddev + D.4710;
  [corr_ref.c : 47:15] D.4709 = (long unsigned int) j;
  [corr_ref.c : 47:15] D.4710 = D.4709 * 8;
  [corr_ref.c : 47:15] D.4721 = stddev + D.4710;
  [corr_ref.c : 47:15] D.4722 = [corr_ref.c : 47] *D.4721;
  [corr_ref.c : 47:15] D.4727 = sqrt (D.4722);
  [corr_ref.c : 47:44] [corr_ref.c : 47] *D.4721 = D.4727;
  [corr_ref.c : 48:11] D.4709 = (long unsigned int) j;
  [corr_ref.c : 48:11] D.4710 = D.4709 * 8;
  [corr_ref.c : 48:11] D.4721 = stddev + D.4710;
  [corr_ref.c : 48:23] D.4709 = (long unsigned int) j;
  [corr_ref.c : 48:23] D.4710 = D.4709 * 8;
  [corr_ref.c : 48:23] D.4721 = stddev + D.4710;
  [corr_ref.c : 48:23] D.4722 = [corr_ref.c : 48] *D.4721;
  [corr_ref.c : 48:50] if (D.4722 <= 4.999999888241291046142578125e-3) goto <D.4729>; else goto <D.4730>;
  <D.4729>:
  [corr_ref.c : 48:50] iftmp.0 = 1.0e+0;
  goto <D.4731>;
  <D.4730>:
  [corr_ref.c : 48:49] D.4709 = (long unsigned int) j;
  [corr_ref.c : 48:49] D.4710 = D.4709 * 8;
  [corr_ref.c : 48:49] D.4721 = stddev + D.4710;
  [corr_ref.c : 48:50] iftmp.0 = [corr_ref.c : 48] *D.4721;
  <D.4731>:
  [corr_ref.c : 48:50] [corr_ref.c : 48] *D.4721 = iftmp.0;
  [corr_ref.c : 37:4] j = j + 1;
  <D.4672>:
  [corr_ref.c : 37:4] if (j <= 255) goto <D.4671>; else goto <D.4665>;
  <D.4665>:
  [corr_ref.c : 52:14] i = 0;
  [corr_ref.c : 52:4] goto <D.4680>;
  <D.4679>:
  [corr_ref.c : 54:13] j = 0;
  [corr_ref.c : 54:3] goto <D.4678>;
  <D.4677>:
  [corr_ref.c : 56:26] D.4713 = i * 256;
  [corr_ref.c : 56:26] D.4714 = D.4713 + j;
  [corr_ref.c : 56:26] D.4715 = (long unsigned int) D.4714;
  [corr_ref.c : 56:26] D.4716 = D.4715 * 8;
  [corr_ref.c : 56:26] D.4717 = data + D.4716;
  [corr_ref.c : 56:26] D.4713 = i * 256;
  [corr_ref.c : 56:26] D.4714 = D.4713 + j;
  [corr_ref.c : 56:26] D.4715 = (long unsigned int) D.4714;
  [corr_ref.c : 56:26] D.4716 = D.4715 * 8;
  [corr_ref.c : 56:26] D.4717 = data + D.4716;
  [corr_ref.c : 56:26] D.4718 = [corr_ref.c : 56] *D.4717;
  [corr_ref.c : 56:25] D.4709 = (long unsigned int) j;
  [corr_ref.c : 56:25] D.4710 = D.4709 * 8;
  [corr_ref.c : 56:25] D.4711 = mean + D.4710;
  [corr_ref.c : 56:25] D.4712 = [corr_ref.c : 56] *D.4711;
  [corr_ref.c : 56:26] D.4723 = D.4718 - D.4712;
  [corr_ref.c : 56:26] [corr_ref.c : 56] *D.4717 = D.4723;
  [corr_ref.c : 57:45] D.4713 = i * 256;
  [corr_ref.c : 57:45] D.4714 = D.4713 + j;
  [corr_ref.c : 57:45] D.4715 = (long unsigned int) D.4714;
  [corr_ref.c : 57:45] D.4716 = D.4715 * 8;
  [corr_ref.c : 57:45] D.4717 = data + D.4716;
  [corr_ref.c : 57:45] D.4713 = i * 256;
  [corr_ref.c : 57:45] D.4714 = D.4713 + j;
  [corr_ref.c : 57:45] D.4715 = (long unsigned int) D.4714;
  [corr_ref.c : 57:45] D.4716 = D.4715 * 8;
  [corr_ref.c : 57:45] D.4717 = data + D.4716;
  [corr_ref.c : 57:45] D.4718 = [corr_ref.c : 57] *D.4717;
  [corr_ref.c : 57:42] D.4709 = (long unsigned int) j;
  [corr_ref.c : 57:42] D.4710 = D.4709 * 8;
  [corr_ref.c : 57:42] D.4721 = stddev + D.4710;
  [corr_ref.c : 57:42] D.4722 = [corr_ref.c : 57] *D.4721;
  [corr_ref.c : 57:33] D.4732 = D.4722 * 1.792822355951642975924187339842319488525390625e+3;
  [corr_ref.c : 57:45] D.4733 = D.4718 / D.4732;
  [corr_ref.c : 57:45] [corr_ref.c : 57] *D.4717 = D.4733;
  [corr_ref.c : 54:3] j = j + 1;
  <D.4678>:
  [corr_ref.c : 54:3] if (j <= 255) goto <D.4677>; else goto <D.4675>;
  <D.4675>:
  [corr_ref.c : 52:4] i = i + 1;
  <D.4680>:
  [corr_ref.c : 52:4] if (i <= 255) goto <D.4679>; else goto <D.4673>;
  <D.4673>:
  [corr_ref.c : 62:15] j1 = 0;
  [corr_ref.c : 62:4] goto <D.4692>;
  <D.4691>:
  [corr_ref.c : 64:14] D.4734 = j1 * 257;
  [corr_ref.c : 64:17] D.4735 = (long unsigned int) D.4734;
  [corr_ref.c : 64:17] D.4736 = D.4735 * 8;
  [corr_ref.c : 64:17] D.4737 = symmat + D.4736;
  [corr_ref.c : 64:25] [corr_ref.c : 64] *D.4737 = 1.0e+0;
  [corr_ref.c : 66:17] j2 = j1 + 1;
  [corr_ref.c : 66:3] goto <D.4690>;
  <D.4689>:
  [corr_ref.c : 68:15] D.4738 = j1 * 256;
  [corr_ref.c : 68:17] D.4739 = D.4738 + j2;
  [corr_ref.c : 68:20] D.4740 = (long unsigned int) D.4739;
  [corr_ref.c : 68:20] D.4741 = D.4740 * 8;
  [corr_ref.c : 68:20] D.4742 = symmat + D.4741;
  [corr_ref.c : 68:28] [corr_ref.c : 68] *D.4742 = 0.0;
  [corr_ref.c : 70:16] i = 0;
  [corr_ref.c : 70:6] goto <D.4688>;
  <D.4687>:
  [corr_ref.c : 72:56] D.4738 = j1 * 256;
  [corr_ref.c : 72:56] D.4739 = D.4738 + j2;
  [corr_ref.c : 72:56] D.4740 = (long unsigned int) D.4739;
  [corr_ref.c : 72:56] D.4741 = D.4740 * 8;
  [corr_ref.c : 72:56] D.4742 = symmat + D.4741;
  [corr_ref.c : 72:56] D.4738 = j1 * 256;
  [corr_ref.c : 72:56] D.4739 = D.4738 + j2;
  [corr_ref.c : 72:56] D.4740 = (long unsigned int) D.4739;
  [corr_ref.c : 72:56] D.4741 = D.4740 * 8;
  [corr_ref.c : 72:56] D.4742 = symmat + D.4741;
  [corr_ref.c : 72:56] D.4743 = [corr_ref.c : 72] *D.4742;
  [corr_ref.c : 72:34] D.4713 = i * 256;
  [corr_ref.c : 72:36] D.4744 = D.4713 + j1;
  [corr_ref.c : 72:39] D.4745 = (long unsigned int) D.4744;
  [corr_ref.c : 72:39] D.4746 = D.4745 * 8;
  [corr_ref.c : 72:39] D.4747 = data + D.4746;
  [corr_ref.c : 72:39] D.4748 = [corr_ref.c : 72] *D.4747;
  [corr_ref.c : 72:49] D.4713 = i * 256;
  [corr_ref.c : 72:51] D.4749 = D.4713 + j2;
  [corr_ref.c : 72:54] D.4750 = (long unsigned int) D.4749;
  [corr_ref.c : 72:54] D.4751 = D.4750 * 8;
  [corr_ref.c : 72:54] D.4752 = data + D.4751;
  [corr_ref.c : 72:54] D.4753 = [corr_ref.c : 72] *D.4752;
  [corr_ref.c : 72:41] D.4754 = D.4748 * D.4753;
  [corr_ref.c : 72:56] D.4755 = D.4743 + D.4754;
  [corr_ref.c : 72:56] [corr_ref.c : 72] *D.4742 = D.4755;
  [corr_ref.c : 70:6] i = i + 1;
  <D.4688>:
  [corr_ref.c : 70:6] if (i <= 255) goto <D.4687>; else goto <D.4685>;
  <D.4685>:
  [corr_ref.c : 75:15] D.4756 = j2 * 256;
  [corr_ref.c : 75:17] D.4757 = D.4756 + j1;
  [corr_ref.c : 75:20] D.4758 = (long unsigned int) D.4757;
  [corr_ref.c : 75:20] D.4759 = D.4758 * 8;
  [corr_ref.c : 75:20] D.4760 = symmat + D.4759;
  [corr_ref.c : 75:33] D.4738 = j1 * 256;
  [corr_ref.c : 75:35] D.4739 = D.4738 + j2;
  [corr_ref.c : 75:38] D.4740 = (long unsigned int) D.4739;
  [corr_ref.c : 75:38] D.4741 = D.4740 * 8;
  [corr_ref.c : 75:38] D.4742 = symmat + D.4741;
  [corr_ref.c : 75:38] D.4743 = [corr_ref.c : 75] *D.4742;
  [corr_ref.c : 75:39] [corr_ref.c : 75] *D.4760 = D.4743;
  [corr_ref.c : 66:3] j2 = j2 + 1;
  <D.4690>:
  [corr_ref.c : 66:3] if (j2 <= 255) goto <D.4689>; else goto <D.4683>;
  <D.4683>:
  [corr_ref.c : 62:4] j1 = j1 + 1;
  <D.4692>:
  [corr_ref.c : 62:4] if (j1 <= 254) goto <D.4691>; else goto <D.4681>;
  <D.4681>:
  [corr_ref.c : 79:24] D.4761 = symmat + 524280;
  [corr_ref.c : 79:32] [corr_ref.c : 79] *D.4761 = 1.0e+0;
  [corr_ref.c : 80:1] return;
}



;; Function int main() (main, funcdef_no=49, decl_uid=4693, symbol_order=49)

int main() ()
{
  double * stddev;
  double * mean;
  double * symmat;
  double * data;
  int i;
  int D.4772;
  double D.4771;
  double * D.4770;
  double D.4769;
  double D.4768;
  int D.4767;
  double * D.4766;
  long unsigned int D.4765;
  long unsigned int D.4764;
  long unsigned int D.4763;
  long unsigned int D.4762;

  [corr_ref.c : 86:51] D.4762 = 524288;
  [corr_ref.c : 86:51] data = malloc (D.4762);
  [corr_ref.c : 87:53] D.4762 = 524288;
  [corr_ref.c : 87:53] symmat = malloc (D.4762);
  [corr_ref.c : 88:49] D.4763 = 2048;
  [corr_ref.c : 88:49] mean = malloc (D.4763);
  [corr_ref.c : 89:51] D.4763 = 2048;
  [corr_ref.c : 89:51] stddev = malloc (D.4763);
  [corr_ref.c : 91:13] srand (5497);
  [corr_ref.c : 92:15] i = 0;
  [corr_ref.c : 92:5] goto <D.4703>;
  <D.4702>:
  [corr_ref.c : 93:15] D.4764 = (long unsigned int) i;
  [corr_ref.c : 93:15] D.4765 = D.4764 * 8;
  [corr_ref.c : 93:15] D.4766 = data + D.4765;
  [corr_ref.c : 93:31] D.4767 = rand ();
  [corr_ref.c : 93:33] D.4768 = (double) D.4767;
  [corr_ref.c : 93:33] D.4769 = D.4768 / 2.147483647e+9;
  [corr_ref.c : 93:50] [corr_ref.c : 93] *D.4766 = D.4769;
  [corr_ref.c : 92:5] i = i + 1;
  <D.4703>:
  [corr_ref.c : 92:5] if (i <= 65535) goto <D.4702>; else goto <D.4700>;
  <D.4700>:
  [corr_ref.c : 95:44] correlation (data, mean, stddev, symmat);
  [corr_ref.c : 97:12] i = 0;
  [corr_ref.c : 97:2] goto <D.4707>;
  <D.4706>:
  [corr_ref.c : 98:28] D.4764 = (long unsigned int) i;
  [corr_ref.c : 98:28] D.4765 = D.4764 * 8;
  [corr_ref.c : 98:28] D.4770 = symmat + D.4765;
  [corr_ref.c : 98:29] D.4771 = [corr_ref.c : 98] *D.4770;
  [corr_ref.c : 98:30] printf ("%.15f,", D.4771);
  [corr_ref.c : 97:2] i = i + 1;
  <D.4707>:
  [corr_ref.c : 97:2] if (i <= 65535) goto <D.4706>; else goto <D.4704>;
  <D.4704>:
  [corr_ref.c : 101:9] D.4772 = 0;
  [corr_ref.c : 101:9] goto <D.4773>;
  [corr_ref.c : 102:1] D.4772 = 0;
  [corr_ref.c : 102:1] goto <D.4773>;
  <D.4773>:
  return D.4772;
}



;; Function int printf(const char*, ...) (<unset-asm-name>, funcdef_no=16, decl_uid=955, symbol_order=16)

int printf(const char*, ...) (const char * restrict __fmt)
{
  int D.4776;
  int D.4774;

  [/usr/include/x86_64-linux-gnu/bits/stdio2.h : 104:72] D.4776 = __printf_chk (1, __fmt, __builtin_va_arg_pack ());
  [/usr/include/x86_64-linux-gnu/bits/stdio2.h : 104:72] D.4774 = D.4776;
  [/usr/include/x86_64-linux-gnu/bits/stdio2.h : 104:72] goto <D.4775>;
  <D.4775>:
  [/usr/include/x86_64-linux-gnu/bits/stdio2.h : 104:72] return D.4774;
}


