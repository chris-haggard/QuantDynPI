subroutine dipole_wrapper(q, ndim, natoms, nmol, mu) 
   implicit none
   integer, intent(in) :: ndim, natoms, nmol
   integer :: i_nmol
   double precision, dimension(ndim, natoms, nmol), intent(in) :: q
   double precision, dimension(ndim, nmol), intent(out) :: mu
   double precision, dimension(natoms, ndim) :: qT
   double precision, dimension(3) :: mu_temp

   do i_nmol = 1, nmol
      qT = reshape(q(:, :, i_nmol), shape=(/natoms, ndim/), order=[2, 1])
      ! convert atomic units (Bohr) to angstrom
      qT = qT * 0.52917715
      call DIPOLE_XY3(qT, mu_temp)
      mu(:, i_nmol) = mu_temp
   end do
end subroutine dipole_wrapper

subroutine DIPOLE_XY3(q, mu_i)

   ! Computes dipole moment for SINGLE MOLECULE

   implicit none

!    *****
!    File downloaded and adapted from https://pubs.acs.org/doi/abs/10.1021/jp9029425 (supporting info)
!    *****

!    This program computes the dipole moment of XY3 molecules using the FUNCTIONS DMS_A and dipol_xy see below.
!    The dipole moment \vec{mu} is given by the three symmetrically adapted projections of \vec{mu} onto
!    1) the projections \bar{mu}_i (i=1,2,3) on the three molecular bonds X-Y1, X-Y2, and X-Y3:
!    \bar{mu}_i = (\vec{mu}\cdot \vec{r_i}/|r_i| ),
!    where (a\cdot b) denotes a scalar product of the vectors a and b and
!    r_i is a vector, coinciding with the bond X-Y_i and pointing to the atom Y_i, and
!    2) the projection onto the trisector unitary vector \bar{n_4} = \bar{b}/|\bar{b}|, where
!    \bar{b} = (e1 x e2) + (e2 x e3) + (e3 x e1).
!    Thus the three independed components are
!    mu_A" = (\bar{mu},n_4)
!    mu_Ea = 1/sqrt(6) [2 \bar{mu}_1 - \bar{mu}_2 -  \bar{mu}_3]
!    mu_Eb = 1/sqrt(2) [               \bar{mu}_2 -  \bar{mu}_3]
!    The program will compute the Cartesain components of the dipole moment employing the LAPACK
!    procedure dgelss. The input geometries have to be given in the same Cartesian coordinate system
!    used to represent the the dipole moment.
!    For the details on the definition of \bar{mu}_i see the paper.
!
!    The Cartesian coordinates must be given in the following order (e.g. for NH3):
!    x_H1, y_H1, z_H1, x_H2, y_H2, z_H2, x_H2, y_H3, z_H3, x_N, y_N, z_N
!    The line positions in the input file are important.
!    The input files contain weight factors for each dipole moment parameter.
!    These are redundant here and are coming from the fitting procedure.
!    The relative positions of the weights are important in the input
!    file, as well as positions of the parameter labels and their values. They must stay
!    within the right sections, see either the input file examples or
!    the corresponding READ command. We hope the program is simple and self-explanatory.
!

!
! Parameters:
   !
   ! number of parameters
   !
   integer, parameter          ::  parmax = 261, enermax = 50000, parmax1 = 149, parmax2 = 112
   !
   ! where the output goes to
   !
   integer, parameter          :: f_inp = 5, f_out = 6, f_err = 0 !!! CHANGE THESE !!!
   !
   ! constants
   !
   double precision, parameter ::pi = 3.141592653589793
   !
   integer                    ::  lwork = 50
   !
   !
   integer           :: npts, i
   !
   character(len=300):: title(4), longlabel ! input title of the job from the first four input lines
   character(len=10) :: parnam(parmax) ! Names of the parameters, as they appear in the input file
   character(len=10) :: label
   !
   double precision  :: local(7)
   integer           :: alloc
   logical           :: yes
   integer           :: ivartmp, ivar(parmax), order, rank0, ierror, ix, iy, iz, n, x_singular
   double precision  :: paramtmp, DMS_A, dipol_xy, charge, EC, Xshift, VCPRM !,mu(3)
   double precision  :: r1, r2, r3, a1, a2, a3, dip(3), edip(3, 1), wspace(50), rmat(3, 3), tmat(4, 3), dip_cart(3)
   double precision  :: tsing(3), bmat(3), f_t, amat(3, 3), mass(4), total_mass, n3(1:3), v12(3), v23(3), v31(3)
   double precision, dimension(4, 3), intent(in)  :: q
   double precision, dimension(4, 3) :: x
   double precision, dimension(3), intent(out)  :: mu_i

   !
   ! some matrices, to be allocated
   !
   double precision :: param(261) = (/ 90.00000000,&
   1.01032000,&
 1.00000000,&
 0.92188119,&
 -14.06001849,&
 69.26186036,&
 -230.90626922,&
 305.66375906,&
 0.51593115,&
 3.60708303,&
 -9.54643796,&
 22.03710110,&
 0.00000000,&
 -0.28262410,&
 -9.62191637,&
 29.02142418,&
 0.00000000,&
 0.02823652,&
 -2.43741249,&
 18.73964165,&
 -42.30830010,&
 -1.46294571,&
 6.65112189,&
 -33.00489064,&
 92.37597965,&
 0.60718426,&
 -1.33929138,&
 -1.22189129,&
 0.00000000,&
 0.03765889,&
 0.52055287,&
 0.00000000,&
 -3.66138376,&
 0.17255875,&
 -6.96917148,&
 19.43796726,&
 0.00000000,&
 -0.47324141,&
 -1.30556902,&
 6.86812708,&
 0.00000000,&
 0.46613709,&
 0.62761017,&
 0.00000000,&
 0.00000000,&
 -0.14027826,&
 7.64352250,&
 -29.99059928,&
 0.00000000,&
 -0.27446296,&
 3.94820596,&
 -14.35025570,&
 0.00000000,&
 -0.04502158,&
 -8.69582649,&
 27.58902920,&
 0.00000000,&
 0.52178586,&
 0.00000000,&
 -9.05550318,&
 41.52225408,&
 -0.11603489,&
 -1.61141411,&
 22.89927944,&
 -80.35994820,&
 -0.01330459,&
 0.00000000,&
 16.95336808,&
 -55.77657424,&
 -0.58672227,&
 5.30406827,&
 -22.84402543,&
 0.00000000,&
 0.10997790,&
 8.58450912,&
 -54.31491540,&
 152.08712646,&
 -1.08732105,&
 -19.18866016,&
 88.63685895,&
 -1.42184489,&
 4.99160778,&
 0.00000000,&
 0.12408251,&
 0.00000000,&
 -10.10344083,&
 0.22684329,&
 4.65663229,&
 0.00000000,&
 -0.89164046,&
 7.37320036,&
 0.00000000,&
 1.99748633,&
 -10.00351645,&
 0.00000000,&
 -1.51584485,&
 13.21676137,&
 0.00000000,&
 -1.90318587,&
 29.03051280,&
 -122.77069471,&
 1.71215480,&
 -25.09690013,&
 78.17055375,&
 -0.38682477,&
 1.58662250,&
 0.00000000,&
 -0.45889370,&
 5.61512574,&
 0.00000000,&
 -0.05062053,&
 0.65097014,&
 0.00000000,&
 0.09014489,&
 -0.76133696,&
 0.00000000,&
 -0.20515444,&
 0.00000000,&
 0.00000000,&
 -0.13994813,&
 1.39186098,&
 -3.70301556,&
 1.91174807,&
 -16.43429920,&
 0.00000000,&
 0.00000000,&
 4.57917498,&
 0.00000000,&
 0.14108076,&
 1.84522088,&
 0.00000000,&
 -0.25382677,&
 4.92775328,&
 0.00000000,&
 -1.25736491,&
 7.57134776,&
 0.00000000,&
 0.39202307,&
 -1.19166896,&
 0.00000000,&
 0.00000000,&
 -1.82660133,&
 0.00000000,&
 0.25106039,&
 -9.58982276,&
 0.00000000,&
 0.24700907,&
 -2.20058034,&
 0.00000000,&
 90.00000000,&
 1.01032000,&
 1.00000000,&
 4.56621083,&
 -9.36438932,&
 32.96400671,&
 -80.82339377,&
 91.52934266,&
 0.00000000,&
 0.00000000,&
 0.00000000,&
 -0.26803382,&
 -5.24006003,&
 49.80073884,&
 -205.26327590,&
 308.82022394,&
 0.00000000,&
 0.00000000,&
 0.19095616,&
 -6.83684850,&
 32.91957941,&
 -61.54016406,&
 0.00000000,&
 -1.32075031,&
 8.89793265,&
 -16.74603082,&
 0.00000000,&
 0.00000000,&
 -0.16279142,&
 3.72814534,&
 -11.79646157,&
 0.00000000,&
 0.00000000,&
 -0.58942127,&
 3.17438042,&
 0.00000000,&
 -15.52485631,&
 0.00000000,&
 -1.46509769,&
 2.72413533,&
 0.00000000,&
 0.00000000,&
 -0.36836346,&
 2.66188419,&
 0.00000000,&
 0.00000000,&
 -0.40496479,&
 6.46872939,&
 -32.65988507,&
 0.00000000,&
 -1.75400136,&
 9.13020014,&
 0.00000000,&
 0.00000000,&
 0.62816666,&
 1.94674667,&
 -7.38249853,&
 0.00000000,&
 0.09277087,&
 2.03150317,&
 -7.20738644,&
 0.00000000,&
 -0.22522596,&
 1.00894414,&
 -11.95331943,&
 0.00000000,&
 0.46031063,&
 8.47191043,&
 -30.43193372,&
 0.00000000,&
 -0.68840781,&
 0.00000000,&
 0.00000000,&
 -1.72240320,&
 9.24077045,&
 0.00000000,&
 0.00000000,&
 9.06932779,&
 0.00000000,&
 0.00000000,&
 3.72705288,&
 0.00000000,&
 -0.88609260,&
 0.00000000,&
 0.00000000,&
 1.00173526,&
 -3.22432030,&
 0.00000000,&
 0.72260912,&
 -3.66971995,&
 0.00000000,&
 -0.24677296,&
 2.54157759,&
 0.00000000,&
 -0.84614401,&
 0.00000000,&
 0.00000000,&
 0.12260292,&
 -1.90027108,&
 0.00000000,&
 0.26514361,&
 -4.40308056,&
 0.00000000,&
 0.08067096,&
 0.26421581,&
 0.00000000,&
 0.26715485,&
 -5.43983594,&
 0.00000000,&
 0.31001469,&
 -5.85896655,&
 0.00000000/)

!read (longlabel,*) x(1,1:3),x(2,1:3),x(3,1:3),x(4,1:3)

!
   x(1, :) = q(1, :) - q(4, :)
   x(2, :) = q(2, :) - q(4, :)
   x(3, :) = q(3, :) - q(4, :)
!
   local(1) = sqrt(sum(x(1, :)**2))
   local(2) = sqrt(sum(x(2, :)**2))
   local(3) = sqrt(sum(x(3, :)**2))
!
   local(4) = acos(sum(x(2, :)*x(3, :))/(local(2)*local(3)))
   local(5) = acos(sum(x(1, :)*x(3, :))/(local(1)*local(3)))
   local(6) = acos(sum(x(2, :)*x(1, :))/(local(2)*local(1)))
!
! Value of the dipole moment fucntion at the current geometry
!
   tmat(1, :) = x(1, :)/local(1)
   tmat(2, :) = x(2, :)/local(2)
   tmat(3, :) = x(3, :)/local(3)
!
   call vector_product(x(1, :)/local(1), x(2, :)/local(2), v12)
   call vector_product(x(2, :)/local(2), x(3, :)/local(3), v23)
   call vector_product(x(3, :)/local(3), x(1, :)/local(1), v31)
!
   n3 = v12 + v23 + v31
!
   n3 = n3/sqrt(sum(n3(:)**2))
!
   local(7) = acos(sum(x(1, :)*n3(:))/local(1))
!
   tmat(4, :) = n3(:)
!
   edip(1, 1) = dipol_xy(1, parmax1, param(1:parmax1), local)
   edip(2, 1) = dipol_xy(2, parmax1, param(1:parmax1), local)
   edip(3, 1) = DMS_A(parmax2, param(parmax1 + 1:parmax), local)
!
   rmat(1, :) = 1.d0/sqrt(6.d0)*(2.0d0*tmat(1, :) - tmat(2, :) - tmat(3, :))
   rmat(2, :) = 1.d0/sqrt(2.d0)*(tmat(2, :) - tmat(3, :))
   rmat(3, :) = tmat(4, :)
!
   call dgelss(3, 3, 1, rmat, 3, edip(1:3, 1), 3, tsing, 10.0d0*spacing(1.0d0), rank0, wspace, lwork, ierror)
!
   mu_i = edip(:, 1)

end subroutine DIPOLE_XY3

double precision function dipol_xy(ix,parmax,param,local)
   
    integer,intent(in)          ::  ix,parmax
    double precision,intent(in) ::  param(parmax)
    double precision,intent(in) ::  local(7)
   
    double precision         ::  r14,r24,r34,alpha1,alpha2,alpha3
    double precision         ::  y1,y2,y3
    double precision         ::  s4a,s4b,alpha,rho
   
    double precision         ::  rhoedg
   
    double precision         ::  d4,sinrho,drho,beta,b0
   
    double precision         :: rhoe     ,re14, rhobar, factor
   
    double precision          &
       FEA1     , F0A1     , F1A1     , F2A1     , F3A1  ,F4A1,  &
       FEA4     , F0A4     , F1A4     , F2A4     , F3A4  ,F4A4,  &
       FEA11    , F0A11    , F1A11    , F2A11    , F3A11 ,       &
       FEA12    , F0A12    , F1A12    , F2A12    , F3A12 ,       &
       FEA14    , F0A14    , F1A14    , F2A14    , F3A14 ,       &
       FEA24    , F0A24    , F1A24    , F2A24    , F3A24 ,       &
       FEA44    , F0A44    , F1A44    , F2A44    , F3A44 ,       &
       FEA111   , F0A111   , F1A111   , F2A111   , F3A111,       &
       FEA112   , F0A112   , F1A112   , F2A112   , F3A112,       &
       FEA122   , F0A122   , F1A122   , F2A122   , F3A122,       &
       FEA114   , F0A114   , F1A114   , F2A114   , F3A114,       &
       FEA124   , F0A124   , F1A124   , F2A124   , F3A124,       &
       FEA224   , F0A224   , F1A224   , F2A224   , F3A224,       &
       FEA144   , F0A144   , F1A144   , F2A144   , F3A144,       &
       FEA244   , F0A244   , F1A244   , F2A244   , F3A244,       &
       FEA444   , F0A444   , F1A444   , F2A444   , F3A444,       &
       FEA135   , F0A135   , F1A135   , F2A135   , F3A135,       &
       FEA155   , F0A155   , F1A155   , F2A155   , F3A155,       &
       FEA1111  , F0A1111  , F1A1111  , F2A1111  ,               &
       FEA1112  , F0A1112  , F1A1112  , F2A1112  ,               &
       FEA1122  , F0A1122  , F1A1122  , F2A1122  ,               &
       FEA1222  , F0A1222  , F1A1222  , F2A1222  ,               &
       FEA1123  , F0A1123  , F1A1123  , F2A1123  ,               &
       FEA1114  , F0A1114  , F1A1114  , F2A1114  ,               &
       FEA1124  , F0A1124  , F1A1124  , F2A1124  ,               &
       FEA1224  , F0A1224  , F1A1224  , F2A1224  ,               &
       FEA2224  , F0A2224  , F1A2224  , F2A2224  ,               &
       FEA1234  , F0A1234  , F1A1234  , F2A1234  ,               &
       FEA1144  , F0A1144  , F1A1144  , F2A1144  ,               &
       FEA1244  , F0A1244  , F1A1244  , F2A1244  ,               &
       FEA1444  , F0A1444  , F1A1444  , F2A1444  ,               &
       FEA2444  , F0A2444  , F1A2444  , F2A2444  ,               &
       FEA4444  , F0A4444  , F1A4444  , F2A4444  ,               &
       FEA1125  , F0A1125  , F1A1125  , F2A1125  ,               &
       FEA1225  , F0A1225  , F1A1225  , F2A1225  ,               &
       FEA1245  , F0A1245  , F1A1245  , F2A1245  ,               &
       FEA2445  , F0A2445  , F1A2445  , F2A2445  ,               &
       FEA1155  , F0A1155  , F1A1155  , F2A1155  ,               &
       FEA1255  , F0A1255  , F1A1255  , F2A1255  ,               &
       FEA3355  , F0A3355  , F1A3355  , F2A3355  ,               &
       FEA1455  , F0A1455  , F1A1455  , F2A1455  ,               &
       FEA4455  , F0A4455  , F1A4455  , F2A4455  ,               &
       FEA11111 , F0A11111 , F1A11111 ,                          &
       FEA11112 , F0A11112 , F1A11112 ,                          &
       FEA11122 , F0A11122 , F1A11122 ,                          &
       FEA11123 , F0A11123 , F1A11123 ,                          &
       FEA11223 , F0A11223 , F1A11223 ,                          &
       FEA11114 , F0A11114 , F1A11114 ,                          &
       FEA11124 , F0A11124 , F1A11124 ,                          &
       FEA11224 , F0A11224 , F1A11224 ,                          &
       FEA11234 , F0A11234 , F1A11234 ,                          &
       FEA22344 , F0A22344 , F1A22344 ,                          &
       FEA11135 , F0A11135 , F1A11135 ,                          &
       FEA22235 , F0A22235 , F1A22235 ,                          &
       FEA11245 , F0A11245 , F1A11245 ,                          &
       FEA12245 , F0A12245 , F1A12245 ,                          &
       FEA22245 , F0A22245 , F1A22245 ,                          &
       FEA11155 , F0A11155 , F1A11155 ,                          &
       FEA11255 , F0A11255 , F1A11255 ,                          &
       FEA12255 , F0A12255 , F1A12255 ,                          &
       FEA22255 , F0A22255 , F1A22255 ,                          &
       FEA12355 , F0A12355 , F1A12355 ,                          &
       FEA11455 , F0A11455 , F1A11455 ,                          &
       FEA12455 , F0A12455 , F1A12455 ,                          &
       FEA22455 , F0A22455 , F1A22455 ,                          &
       FEA23455 , F0A23455 , F1A23455 ,                          &
       FEA24455 , F0A24455 , F1A24455 ,                          &
       FEA12555 , F0A12555 , F1A12555 ,                          &
       FEA22555 , F0A22555 , F1A22555 ,                          &
       FEA24555 , F0A24555 , F1A24555 ,                          &
       FEA15555 , F0A15555 , F1A15555 ,                          &
       FEA45555 , F0A45555 , F1A45555 ,                          &
       FEA111111, F0A111111, F1A111111,                          &
       FEA111112, F0A111112, F1A111112,                          &
       FEA111122, F0A111122, F1A111122,                          &
       FEA122333, F0A122333, F1A122333,                          &
       FEA123333, F0A123333, F1A123333,                          &
       FEA122334, F0A122334, F1A122334,                          &
       FEA123334, F0A123334, F1A123334,                          &
       FEA223334, F0A223334, F1A223334,                          &
       FEA133334, F0A133334, F1A133334,                          &
       FEA233334, F0A233334, F1A233334,                          &
       FEA333334, F0A333334, F1A333334,                          &
       FEA111344, F0A111344, F1A111344,                          &
       FEA112344, F0A112344, F1A112344,                          &
       FEA223344, F0A223344, F1A223344,                          &
       FEA133344, F0A133344, F1A133344,                          &
       FEA233344, F0A233344, F1A233344,                          &
       FEA333344, F0A333344, F1A333344,                          &
       FEA122444, F0A122444, F1A122444,                          &
       FEA222444, F0A222444, F1A222444,                          &
       FEA113444, F0A113444, F1A113444,                          &
       FEA123444, F0A123444, F1A123444,                          &
       FEA124444, F0A124444, F1A124444,                          &
       FEA444444, F0A444444, F1A444444,                          &
       FEA233335, F0A233335, F1A233335,                          &
       FEA233345, F0A233345, F1A233345,                          &
       FEA113445, F0A113445, F1A113445,                          &
       FEA133445, F0A133445, F1A133445,                          &
       FEA233445, F0A233445, F1A233445,                          &
       FEA235555, F0A235555, F1A235555,                          &
       FEA335555, F0A335555, F1A335555,                          &
       FEA145555, F0A145555, F1A145555,                          &
       HEA11112 , H0A11112 , H1A11112 ,                          &
       HEA11122 , H0A11122 , H1A11122 ,                          &
       HEA11244 , H0A11244 , H1A11244 ,                          &
       HEA13444 , H0A13444 , H1A13444 ,                          &
       HEA33444 , H0A33444 , H1A33444 ,                          &
       HEA12335 , H0A12335 , H1A12335 ,                          &
       HEA22335 , H0A22335 , H1A22335 ,                          &
       HEA13335 , H0A13335 , H1A13335 ,                          &
       HEA33335 , H0A33335 , H1A33335 ,                          &
       HEA14445 , H0A14445 , H1A14445 ,                          &
       HEA34555 , H0A34555 , H1A34555 ,                          &
       HEA55555 , H0A55555 , H1A55555 ,                          &
       HEA111333, H0A111333, H1A111333,                          &
       HEA122333, H0A122333, H1A122333,                          &
       HEA223333, H0A223333, H1A223333,                          &
       HEA233333, H0A233333, H1A233333,                          &
       HEA111334, H0A111334, H1A111334,                          &
       HEA113334, H0A113334, H1A113334,                          &
       HEA223334, H0A223334, H1A223334,                          &
       HEA233334, H0A233334, H1A233334,                          &
       HEA113344, H0A113344, H1A113344,                          &
       HEA123344, H0A123344, H1A123344,                          &
       HEA133344, H0A133344, H1A133344,                          &
       HEA233344, H0A233344, H1A233344,                          &
       HEA133444, H0A133444, H1A133444,                          &
       HEA233444, H0A233444, H1A233444,                          &
       HEA333444, H0A333444, H1A333444,                          &
       HEA122335, H0A122335, H1A122335,                          &
       HEA123335, H0A123335, H1A123335,                          &
       HEA333335, H0A333335, H1A333335,                          &
       HEA111145, H0A111145, H1A111145,                          &
       HEA123345, H0A123345, H1A123345,                          &
       HEA223345, H0A223345, H1A223345,                          &
       HEA234445, H0A234445, H1A234445,                          &
       HEA334445, H0A334445, H1A334445,                          &
       HEA144445, H0A144445, H1A144445,                          &
       HEA344445, H0A344445, H1A344445,                          &
       HEA444445, H0A444445, H1A444445,                          &
       HEA333355, H0A333355, H1A333355,                          &
       HEA333455, H0A333455, H1A333455,                          &
       HEA334455, H0A334455, H1A334455,                          &
       HEA344455, H0A344455, H1A344455,                          &
       HEA222555, H0A222555, H1A222555,                          &
       HEA223555, H0A223555, H1A223555,                          &
       HEA114555, H0A114555, H1A114555,                          &
       HEA234555, H0A234555, H1A234555,                          &
       HEA334555, H0A334555, H1A334555,                          &
       HEA135555, H0A135555, H1A135555,                          &
       HEA345555, H0A345555, H1A345555,                          &
       HEA355555, H0A355555, H1A355555
   
     double precision d56,s1,s2,s3,s4,s5,s6,pi,t4x,t4y,t56x,t56y,cosrho
   
   
   
     !-------------------------------
   
   
       pi=3.141592653589793
   
       rhoedg     = param( 1)
       rhoe=pi*rhoedg/1.8d+02
   
       re14      =  param(  2)
       b0        =  param(  3)**2
       F0A1      =  param(  4)
       F1A1      =  param(  5)
       F2A1      =  param(  6)
       F3A1      =  param(  7)
       F4A1      =  param(  8)
       F0A4      =  param(  9)
       F1A4      =  param( 10)
       F2A4      =  param( 11)
       F3A4      =  param( 12)
       F4A4      =  param( 13)
       F0A11     =  param( 14)
       F1A11     =  param( 15)
       F2A11     =  param( 16)
       F3A11     =  param( 17)
       F0A12     =  param( 18)
       F1A12     =  param( 19)
       F2A12     =  param( 20)
       F3A12     =  param( 21)
       F0A14     =  param( 22)
       F1A14     =  param( 23)
       F2A14     =  param( 24)
       F3A14     =  param( 25)
       F0A24     =  param( 26)
       F1A24     =  param( 27)
       F2A24     =  param( 28)
       F3A24     =  param( 29)
       F0A44     =  param( 30)
       F1A44     =  param( 31)
       F2A44     =  param( 32)
       F3A44     =  param( 33)
       F0A111    =  param( 34)
       F1A111    =  param( 35)
       F2A111    =  param( 36)
       F3A111    =  param( 37)
       F0A112    =  param( 38)
       F1A112    =  param( 39)
       F2A112    =  param( 40)
       F3A112    =  param( 41)
       F0A122    =  param( 42)
       F1A122    =  param( 43)
       F2A122    =  param( 44)
       F3A122    =  param( 45)
       F0A114    =  param( 46)
       F1A114    =  param( 47)
       F2A114    =  param( 48)
       F3A114    =  param( 49)
       F0A124    =  param( 50)
       F1A124    =  param( 51)
       F2A124    =  param( 52)
       F3A124    =  param( 53)
       F0A224    =  param( 54)
       F1A224    =  param( 55)
       F2A224    =  param( 56)
       F3A224    =  param( 57)
       F0A144    =  param( 58)
       F1A144    =  param( 59)
       F2A144    =  param( 60)
       F3A144    =  param( 61)
       F0A244    =  param( 62)
       F1A244    =  param( 63)
       F2A244    =  param( 64)
       F3A244    =  param( 65)
       F0A444    =  param( 66)
       F1A444    =  param( 67)
       F2A444    =  param( 68)
       F3A444    =  param( 69)
       F0A135    =  param( 70)
       F1A135    =  param( 71)
       F2A135    =  param( 72)
       F3A135    =  param( 73)
       F0A155    =  param( 74)
       F1A155    =  param( 75)
       F2A155    =  param( 76)
       F3A155    =  param( 77)
       F0A1111   =  param( 78)
       F1A1111   =  param( 79)
       F2A1111   =  param( 80)
       F0A1112   =  param( 81)
       F1A1112   =  param( 82)
       F2A1112   =  param( 83)
       F0A1122   =  param( 84)
       F1A1122   =  param( 85)
       F2A1122   =  param( 86)
       F0A1222   =  param( 87)
       F1A1222   =  param( 88)
       F2A1222   =  param( 89)
       F0A1123   =  param( 90)
       F1A1123   =  param( 91)
       F2A1123   =  param( 92)
       F0A1114   =  param( 93)
       F1A1114   =  param( 94)
       F2A1114   =  param( 95)
       F0A1124   =  param( 96)
       F1A1124   =  param( 97)
       F2A1124   =  param( 98)
       F0A1224   =  param( 99)
       F1A1224   =  param(100)
       F2A1224   =  param(101)
       F0A2224   =  param(102)
       F1A2224   =  param(103)
       F2A2224   =  param(104)
       F0A1234   =  param(105)
       F1A1234   =  param(106)
       F2A1234   =  param(107)
       F0A1144   =  param(108)
       F1A1144   =  param(109)
       F2A1144   =  param(110)
       F0A1244   =  param(111)
       F1A1244   =  param(112)
       F2A1244   =  param(113)
       F0A1444   =  param(114)
       F1A1444   =  param(115)
       F2A1444   =  param(116)
       F0A2444   =  param(117)
       F1A2444   =  param(118)
       F2A2444   =  param(119)
       F0A4444   =  param(120)
       F1A4444   =  param(121)
       F2A4444   =  param(122)
       F0A1125   =  param(123)
       F1A1125   =  param(124)
       F2A1125   =  param(125)
       F0A1225   =  param(126)
       F1A1225   =  param(127)
       F2A1225   =  param(128)
       F0A1245   =  param(129)
       F1A1245   =  param(130)
       F2A1245   =  param(131)
       F0A2445   =  param(132)
       F1A2445   =  param(133)
       F2A2445   =  param(134)
       F0A1155   =  param(135)
       F1A1155   =  param(136)
       F2A1155   =  param(137)
       F0A1255   =  param(138)
       F1A1255   =  param(139)
       F2A1255   =  param(140)
       F0A3355   =  param(141)
       F1A3355   =  param(142)
       F2A3355   =  param(143)
       F0A1455   =  param(144)
       F1A1455   =  param(145)
       F2A1455   =  param(146)
       F0A4455   =  param(147)
       F1A4455   =  param(148)
       F2A4455   =  param(149)
       !
       if (parmax>149) then 
         !
         f0a11111  =  param(150)
         f1a11111  =  param(151)
         f0a11112  =  param(152)
         f1a11112  =  param(153)
         f0a11122  =  param(154)
         f1a11122  =  param(155)
         f0a11123  =  param(156)
         f1a11123  =  param(157)
         f0a11223  =  param(158)
         f1a11223  =  param(159)
         f0a11114  =  param(160)
         f1a11114  =  param(161)
         f0a11124  =  param(162)
         f1a11124  =  param(163)
         f0a11224  =  param(164)
         f1a11224  =  param(165)
         f0a11234  =  param(166)
         f1a11234  =  param(167)
         f0a22344  =  param(168)
         f1a22344  =  param(169)
         f0a11135  =  param(170)
         f1a11135  =  param(171)
         f0a22235  =  param(172)
         f1a22235  =  param(173)
         f0a11245  =  param(174)
         f1a11245  =  param(175)
         f0a12245  =  param(176)
         f1a12245  =  param(177)
         f0a22245  =  param(178)
         f1a22245  =  param(179)
         f0a11155  =  param(180)
         f1a11155  =  param(181)
         f0a11255  =  param(182)
         f1a11255  =  param(183)
         f0a12255  =  param(184)
         f1a12255  =  param(185)
         f0a22255  =  param(186)
         f1a22255  =  param(187)
         f0a12355  =  param(188)
         f1a12355  =  param(189)
         f0a11455  =  param(190)
         f1a11455  =  param(191)
         f0a12455  =  param(192)
         f1a12455  =  param(193)
         f0a22455  =  param(194)
         f1a22455  =  param(195)
         f0a23455  =  param(196)
         f1a23455  =  param(197)
         f0a24455  =  param(198)
         f1a24455  =  param(199)
         f0a12555  =  param(200)
         f1a12555  =  param(201)
         f0a22555  =  param(202)
         f1a22555  =  param(203)
         f0a24555  =  param(204)
         f1a24555  =  param(205)
         f0a15555  =  param(206)
         f1a15555  =  param(207)
         f0a45555  =  param(208)
         f1a45555  =  param(209)
         f0a111111 =  param(210)
         f1a111111 =  param(211)
         f0a111112 =  param(212)
         f1a111112 =  param(213)
         f0a111122 =  param(214)
         f1a111122 =  param(215)
         f0a122333 =  param(216)
         f1a122333 =  param(217)
         f0a123333 =  param(218)
         f1a123333 =  param(219)
         f0a122334 =  param(220)
         f1a122334 =  param(221)
         f0a123334 =  param(222)
         f1a123334 =  param(223)
         f0a223334 =  param(224)
         f1a223334 =  param(225)
         f0a133334 =  param(226)
         f1a133334 =  param(227)
         f0a233334 =  param(228)
         f1a233334 =  param(229)
         f0a333334 =  param(230)
         f1a333334 =  param(231)
         f0a111344 =  param(232)
         f1a111344 =  param(233)
         f0a112344 =  param(234)
         f1a112344 =  param(235)
         f0a223344 =  param(236)
         f1a223344 =  param(237)
         f0a133344 =  param(238)
         f1a133344 =  param(239)
         f0a233344 =  param(240)
         f1a233344 =  param(241)
         f0a333344 =  param(242)
         f1a333344 =  param(243)
         f0a122444 =  param(244)
         f1a122444 =  param(245)
         f0a222444 =  param(246)
         f1a222444 =  param(247)
         f0a113444 =  param(248)
         f1a113444 =  param(249)
         f0a123444 =  param(250)
         f1a123444 =  param(251)
         f0a124444 =  param(252)
         f1a124444 =  param(253)
         f0a444444 =  param(254)
         f1a444444 =  param(255)
         f0a233335 =  param(256)
         f1a233335 =  param(257)
         f0a233345 =  param(258)
         f1a233345 =  param(259)
         f0a113445 =  param(260)
         f1a113445 =  param(261)
         f0a133445 =  param(262)
         f1a133445 =  param(263)
         f0a233445 =  param(264)
         f1a233445 =  param(265)
         f0a235555 =  param(266)
         f1a235555 =  param(267)
         f0a335555 =  param(268)
         f1a335555 =  param(269)
         f0a145555 =  param(270)
         f1a145555 =  param(271)
         h0a11112  =  param(272)
         h1a11112  =  param(273)
         h0a11122  =  param(274)
         h1a11122  =  param(275)
         h0a11244  =  param(276)
         h1a11244  =  param(277)
         h0a13444  =  param(278)
         h1a13444  =  param(279)
         h0a33444  =  param(280)
         h1a33444  =  param(281)
         h0a12335  =  param(282)
         h1a12335  =  param(283)
         h0a22335  =  param(284)
         h1a22335  =  param(285)
         h0a13335  =  param(286)
         h1a13335  =  param(287)
         h0a33335  =  param(288)
         h1a33335  =  param(289)
         h0a14445  =  param(290)
         h1a14445  =  param(291)
         h0a34555  =  param(292)
         h1a34555  =  param(293)
         h0a55555  =  param(294)
         h1a55555  =  param(295)
         h0a111333 =  param(296)
         h1a111333 =  param(297)
         h0a122333 =  param(298)
         h1a122333 =  param(299)
         h0a223333 =  param(300)
         h1a223333 =  param(301)
         h0a233333 =  param(302)
         h1a233333 =  param(303)
         h0a111334 =  param(304)
         h1a111334 =  param(305)
         h0a113334 =  param(306)
         h1a113334 =  param(307)
         h0a223334 =  param(308)
         h1a223334 =  param(309)
         h0a233334 =  param(310)
         h1a233334 =  param(311)
         h0a113344 =  param(312)
         h1a113344 =  param(313)
         h0a123344 =  param(314)
         h1a123344 =  param(315)
         h0a133344 =  param(316)
         h1a133344 =  param(317)
         h0a233344 =  param(318)
         h1a233344 =  param(319)
         h0a133444 =  param(320)
         h1a133444 =  param(321)
         h0a233444 =  param(322)
         h1a233444 =  param(323)
         h0a333444 =  param(324)
         h1a333444 =  param(325)
         h0a122335 =  param(326)
         h1a122335 =  param(327)
         h0a123335 =  param(328)
         h1a123335 =  param(329)
         h0a333335 =  param(330)
         h1a333335 =  param(331)
         h0a111145 =  param(332)
         h1a111145 =  param(333)
         h0a123345 =  param(334)
         h1a123345 =  param(335)
         h0a223345 =  param(336)
         h1a223345 =  param(337)
         h0a234445 =  param(338)
         h1a234445 =  param(339)
         h0a334445 =  param(340)
         h1a334445 =  param(341)
         h0a144445 =  param(342)
         h1a144445 =  param(343)
         h0a344445 =  param(344)
         h1a344445 =  param(345)
         h0a444445 =  param(346)
         h1a444445 =  param(347)
         h0a333355 =  param(348)
         h1a333355 =  param(349)
         h0a333455 =  param(350)
         h1a333455 =  param(351)
         h0a334455 =  param(352)
         h1a334455 =  param(353)
         h0a344455 =  param(354)
         h1a344455 =  param(355)
         h0a222555 =  param(356)
         h1a222555 =  param(357)
         h0a223555 =  param(358)
         h1a223555 =  param(359)
         h0a114555 =  param(360)
         h1a114555 =  param(361)
         h0a234555 =  param(362)
         h1a234555 =  param(363)
         h0a334555 =  param(364)
         h1a334555 =  param(365)
         h0a135555 =  param(366)
         h1a135555 =  param(367)
         h0a345555 =  param(368)
         h1a345555 =  param(369)
         h0a355555 =  param(370)
         h1a355555 =  param(371)
         !
      endif 
       !
       r14    = local(1) ; r24     = local(2) ; r34    = local(3) 
       !
       alpha1 = local(4) ; alpha2 = local(5) ; alpha3 = local(6)
       rhobar = local(7)
       !
       s4a=(2.d0*alpha1-alpha2-alpha3)/dsqrt(6.d0)
       s4b=(alpha2-alpha3)/dsqrt(2.d0)
       !
       alpha=(alpha1+alpha2+alpha3)/3.d0
       !
       if ( 2.d0*sin(alpha*0.5d0)/dsqrt(3.d0).ge.1.0d0 ) then
         sinrho=1.d0
       else
         sinrho = 2.d0*sin(alpha*0.5d0)/dsqrt(3.d0)
       endif
       !
       cosrho =-sqrt(1.d0-sinrho**2)
       !
       factor = sinrho
       !
       !drho= (sin(rhoe)-sin(rhobar))
       !
       drho= (sin(rhoe)-sinrho)
       !
       y1=1.0d0*(r14-re14)*exp(-b0*(r14-re14)**2)
       y2=1.0d0*(r24-re14)*exp(-b0*(r24-re14)**2)
       y3=1.0d0*(r34-re14)*exp(-b0*(r34-re14)**2)
       !
       FEA1      =   F0A1      +   F1A1     *drho   +   F2A1      *drho**2  +   F3A1     *drho**3+   F4A1 *drho**4
       FEA4      =   F0A4      +   F1A4     *drho   +   F2A4      *drho**2  +   F3A4     *drho**3+   F4A4 *drho**4
       FEA11     =   F0A11     +   F1A11    *drho   +   F2A11     *drho**2  +   F3A11    *drho**3
       FEA12     =   F0A12     +   F1A12    *drho   +   F2A12     *drho**2  +   F3A12    *drho**3
       FEA14     =   F0A14     +   F1A14    *drho   +   F2A14     *drho**2  +   F3A14    *drho**3
       FEA24     =   F0A24     +   F1A24    *drho   +   F2A24     *drho**2  +   F3A24    *drho**3
       FEA44     =   F0A44     +   F1A44    *drho   +   F2A44     *drho**2  +   F3A44    *drho**3
       FEA111    =   F0A111    +   F1A111   *drho   +   F2A111    *drho**2  +   F3A111   *drho**3
       FEA112    =   F0A112    +   F1A112   *drho   +   F2A112    *drho**2  +   F3A112   *drho**3
       FEA122    =   F0A122    +   F1A122   *drho   +   F2A122    *drho**2  +   F3A122   *drho**3
       FEA114    =   F0A114    +   F1A114   *drho   +   F2A114    *drho**2  +   F3A114   *drho**3
       FEA124    =   F0A124    +   F1A124   *drho   +   F2A124    *drho**2  +   F3A124   *drho**3
       FEA224    =   F0A224    +   F1A224   *drho   +   F2A224    *drho**2  +   F3A224   *drho**3
       FEA144    =   F0A144    +   F1A144   *drho   +   F2A144    *drho**2  +   F3A144   *drho**3
       FEA244    =   F0A244    +   F1A244   *drho   +   F2A244    *drho**2  +   F3A244   *drho**3
       FEA444    =   F0A444    +   F1A444   *drho   +   F2A444    *drho**2  +   F3A444   *drho**3
       FEA135    =   F0A135    +   F1A135   *drho   +   F2A135    *drho**2  +   F3A135   *drho**3
       FEA155    =   F0A155    +   F1A155   *drho   +   F2A155    *drho**2  +   F3A155   *drho**3
       FEA1111   =   F0A1111   +   F1A1111  *drho   +   F2A1111   *drho**2
       FEA1112   =   F0A1112   +   F1A1112  *drho   +   F2A1112   *drho**2
       FEA1122   =   F0A1122   +   F1A1122  *drho   +   F2A1122   *drho**2
       FEA1222   =   F0A1222   +   F1A1222  *drho   +   F2A1222   *drho**2
       FEA1123   =   F0A1123   +   F1A1123  *drho   +   F2A1123   *drho**2
       FEA1114   =   F0A1114   +   F1A1114  *drho   +   F2A1114   *drho**2
       FEA1124   =   F0A1124   +   F1A1124  *drho   +   F2A1124   *drho**2
       FEA1224   =   F0A1224   +   F1A1224  *drho   +   F2A1224   *drho**2
       FEA2224   =   F0A2224   +   F1A2224  *drho   +   F2A2224   *drho**2
       FEA1234   =   F0A1234   +   F1A1234  *drho   +   F2A1234   *drho**2
       FEA1144   =   F0A1144   +   F1A1144  *drho   +   F2A1144   *drho**2
       FEA1244   =   F0A1244   +   F1A1244  *drho   +   F2A1244   *drho**2
       FEA1444   =   F0A1444   +   F1A1444  *drho   +   F2A1444   *drho**2
       FEA2444   =   F0A2444   +   F1A2444  *drho   +   F2A2444   *drho**2
       FEA4444   =   F0A4444   +   F1A4444  *drho   +   F2A4444   *drho**2
       FEA1125   =   F0A1125   +   F1A1125  *drho   +   F2A1125   *drho**2
       FEA1225   =   F0A1225   +   F1A1225  *drho   +   F2A1225   *drho**2
       FEA1245   =   F0A1245   +   F1A1245  *drho   +   F2A1245   *drho**2
       FEA2445   =   F0A2445   +   F1A2445  *drho   +   F2A2445   *drho**2
       FEA1155   =   F0A1155   +   F1A1155  *drho   +   F2A1155   *drho**2
       FEA1255   =   F0A1255   +   F1A1255  *drho   +   F2A1255   *drho**2
       FEA3355   =   F0A3355   +   F1A3355  *drho   +   F2A3355   *drho**2
       FEA1455   =   F0A1455   +   F1A1455  *drho   +   F2A1455   *drho**2
       FEA4455   =   F0A4455   +   F1A4455  *drho   +   F2A4455   *drho**2
       !
       if (parmax>149) then 
         !
         fea11111  =   f0a11111  +   f1a11111 *drho
         fea11112  =   f0a11112  +   f1a11112 *drho
         fea11122  =   f0a11122  +   f1a11122 *drho
         fea11123  =   f0a11123  +   f1a11123 *drho
         fea11223  =   f0a11223  +   f1a11223 *drho
         fea11114  =   f0a11114  +   f1a11114 *drho
         fea11124  =   f0a11124  +   f1a11124 *drho
         fea11224  =   f0a11224  +   f1a11224 *drho
         fea11234  =   f0a11234  +   f1a11234 *drho
         fea22344  =   f0a22344  +   f1a22344 *drho
         fea11135  =   f0a11135  +   f1a11135 *drho
         fea22235  =   f0a22235  +   f1a22235 *drho
         fea11245  =   f0a11245  +   f1a11245 *drho
         fea12245  =   f0a12245  +   f1a12245 *drho
         fea22245  =   f0a22245  +   f1a22245 *drho
         fea11155  =   f0a11155  +   f1a11155 *drho
         fea11255  =   f0a11255  +   f1a11255 *drho
         fea12255  =   f0a12255  +   f1a12255 *drho
         fea22255  =   f0a22255  +   f1a22255 *drho
         fea12355  =   f0a12355  +   f1a12355 *drho
         fea11455  =   f0a11455  +   f1a11455 *drho
         fea12455  =   f0a12455  +   f1a12455 *drho
         fea22455  =   f0a22455  +   f1a22455 *drho
         fea23455  =   f0a23455  +   f1a23455 *drho
         fea24455  =   f0a24455  +   f1a24455 *drho
         fea12555  =   f0a12555  +   f1a12555 *drho
         fea22555  =   f0a22555  +   f1a22555 *drho
         fea24555  =   f0a24555  +   f1a24555 *drho
         fea15555  =   f0a15555  +   f1a15555 *drho
         fea45555  =   f0a45555  +   f1a45555 *drho
         fea111111 =   f0a111111 +   f1a111111*drho
         fea111112 =   f0a111112 +   f1a111112*drho
         fea111122 =   f0a111122 +   f1a111122*drho
         fea122333 =   f0a122333 +   f1a122333*drho
         fea123333 =   f0a123333 +   f1a123333*drho
         fea122334 =   f0a122334 +   f1a122334*drho
         fea123334 =   f0a123334 +   f1a123334*drho
         fea223334 =   f0a223334 +   f1a223334*drho
         fea133334 =   f0a133334 +   f1a133334*drho
         fea233334 =   f0a233334 +   f1a233334*drho
         fea333334 =   f0a333334 +   f1a333334*drho
         fea111344 =   f0a111344 +   f1a111344*drho
         fea112344 =   f0a112344 +   f1a112344*drho
         fea223344 =   f0a223344 +   f1a223344*drho
         fea133344 =   f0a133344 +   f1a133344*drho
         fea233344 =   f0a233344 +   f1a233344*drho
         fea333344 =   f0a333344 +   f1a333344*drho
         fea122444 =   f0a122444 +   f1a122444*drho
         fea222444 =   f0a222444 +   f1a222444*drho
         fea113444 =   f0a113444 +   f1a113444*drho
         fea123444 =   f0a123444 +   f1a123444*drho
         fea124444 =   f0a124444 +   f1a124444*drho
         fea444444 =   f0a444444 +   f1a444444*drho
         fea233335 =   f0a233335 +   f1a233335*drho
         fea233345 =   f0a233345 +   f1a233345*drho
         fea113445 =   f0a113445 +   f1a113445*drho
         fea133445 =   f0a133445 +   f1a133445*drho
         fea233445 =   f0a233445 +   f1a233445*drho
         fea235555 =   f0a235555 +   f1a235555*drho
         fea335555 =   f0a335555 +   f1a335555*drho
         fea145555 =   f0a145555 +   f1a145555*drho
         hea11112  =   h0a11112  +   h1a11112 *drho
         hea11122  =   h0a11122  +   h1a11122 *drho
         hea11244  =   h0a11244  +   h1a11244 *drho
         hea13444  =   h0a13444  +   h1a13444 *drho
         hea33444  =   h0a33444  +   h1a33444 *drho
         hea12335  =   h0a12335  +   h1a12335 *drho
         hea22335  =   h0a22335  +   h1a22335 *drho
         hea13335  =   h0a13335  +   h1a13335 *drho
         hea33335  =   h0a33335  +   h1a33335 *drho
         hea14445  =   h0a14445  +   h1a14445 *drho
         hea34555  =   h0a34555  +   h1a34555 *drho
         hea55555  =   h0a55555  +   h1a55555 *drho
         hea111333 =   h0a111333 +   h1a111333*drho
         hea122333 =   h0a122333 +   h1a122333*drho
         hea223333 =   h0a223333 +   h1a223333*drho
         hea233333 =   h0a233333 +   h1a233333*drho
         hea111334 =   h0a111334 +   h1a111334*drho
         hea113334 =   h0a113334 +   h1a113334*drho
         hea223334 =   h0a223334 +   h1a223334*drho
         hea233334 =   h0a233334 +   h1a233334*drho
         hea113344 =   h0a113344 +   h1a113344*drho
         hea123344 =   h0a123344 +   h1a123344*drho
         hea133344 =   h0a133344 +   h1a133344*drho
         hea233344 =   h0a233344 +   h1a233344*drho
         hea133444 =   h0a133444 +   h1a133444*drho
         hea233444 =   h0a233444 +   h1a233444*drho
         hea333444 =   h0a333444 +   h1a333444*drho
         hea122335 =   h0a122335 +   h1a122335*drho
         hea123335 =   h0a123335 +   h1a123335*drho
         hea333335 =   h0a333335 +   h1a333335*drho
         hea111145 =   h0a111145 +   h1a111145*drho
         hea123345 =   h0a123345 +   h1a123345*drho
         hea223345 =   h0a223345 +   h1a223345*drho
         hea234445 =   h0a234445 +   h1a234445*drho
         hea334445 =   h0a334445 +   h1a334445*drho
         hea144445 =   h0a144445 +   h1a144445*drho
         hea344445 =   h0a344445 +   h1a344445*drho
         hea444445 =   h0a444445 +   h1a444445*drho
         hea333355 =   h0a333355 +   h1a333355*drho
         hea333455 =   h0a333455 +   h1a333455*drho
         hea334455 =   h0a334455 +   h1a334455*drho
         hea344455 =   h0a344455 +   h1a344455*drho
         hea222555 =   h0a222555 +   h1a222555*drho
         hea223555 =   h0a223555 +   h1a223555*drho
         hea114555 =   h0a114555 +   h1a114555*drho
         hea234555 =   h0a234555 +   h1a234555*drho
         hea334555 =   h0a334555 +   h1a334555*drho
         hea135555 =   h0a135555 +   h1a135555*drho
         hea345555 =   h0a345555 +   h1a345555*drho
         hea355555 =   h0a355555 +   h1a355555*drho
         !
      endif 
   
   
      select case (ix)
      case (1)
   
               s2 = fea122*y1*y3**2+ fea111*y1**3+ (-fea155/2.d0-             &
          fea144/2.d0-fea244)*y3*s4b**2+ fea14*y1*s4a+ fea444*s4a*s4b**2+     &
          fea1122*y1**2*y3**2+ fea2444*y2*s4a**3+ fea1114*y1**3*s4a+          &
          fea4455*s4a**2*s4b**2-fea111*y2**3/2.d0+ fea2444*y3*s4a**3+         &
          fea244*y2*s4a**2+ fea2224*y3**3*s4a+ (-fea155/2.d0-fea144/2.d0-     &
          fea244)*y2*s4b**2+ fea3355*y2**2*s4b**2+ (-fea3355-fea1155/2.d0-    &
          fea1144/2.d0)*y2**2*s4a**2+ fea24*y3*s4a+ fea1112*y1**3*y2+         &
          fea1122*y1**2*y2**2+ fea1144*y1**2*s4a**2+ fea1155*y1**2*s4b**2+    &
          (-fea1222-fea1112)*y2**3*y3+ fea224*y3**2*s4a+ fea12*y1*y3+         &
          fea12*y1*y2+ (5.d0/18.d0*fea1444*sqrt(3.d0)+                        &
          fea1455*sqrt(3.d0)/6.d0-4.d0/9.d0*fea2444*sqrt(3.d0)+               &
          fea2445/3.d0)*y3*s4b**3+ fea1444*y1*s4a**3                          
          
               s3 = fea114*y1**2*s4a+ s2+ sqrt(3.d0)*(fea224-                 &
          fea114)*y2**2*s4b/3.d0+ sqrt(3.d0)*(3.d0*fea1125+                   &
          fea1224*sqrt(3.d0)+ 3.d0*fea1225+                                   &
          fea1124*sqrt(3.d0))*y2**2*y3*s4a/6.d0-fea1111*y3**4/2.d0-           &
          fea1123*y1*y2**2*y3/2.d0-fea1123*y1*y2*y3**2/2.d0-                  &
          fea1225*y1*y3**2*s4b-fea135*y1*y2*s4b-fea2445*y3*s4a**2*s4b+ (-     &
          fea4444-fea4455/3.d0)*s4b**4-fea1245*y1*y3*s4a*s4b+                 &
          fea124*y1*y3*s4a-sqrt(3.d0)*(-fea1114+ fea2224)*y3**3*s4b/3.d0      
               s1 = s3+ fea135*y1*y3*s4b+ fea124*y1*y2*s4a+ sqrt(3.d0)*(-     &
          fea1114+ fea2224)*y2**3*s4b/3.d0+ (-fea1455/2.d0+ fea1444/2.d0+     &
          fea2444)*y3*s4a*s4b**2+ fea4444*s4a**4-fea111*y3**3/2.d0-           &
          fea1111*y2**4/2.d0-sqrt(3.d0)*(4.d0*fea3355-fea1155+                &
          3.d0*fea1144)*y3**2*s4a*s4b/6.d0+ (-fea1455/2.d0+ fea1444/2.d0+     &
          fea2444)*y2*s4a*s4b**2+ (-fea1255/2.d0-3.d0/2.d0*fea1244+           &
          fea1245*sqrt(3.d0)/2.d0)*y2*y3*s4b**2+ fea2445*y2*s4a**2*s4b+       &
          sqrt(3.d0)*(fea24-fea14)*y2*s4b/3.d0-sqrt(3.d0)*(fea24-             &
          fea14)*y3*s4b/3.d0+ (-fea1224*sqrt(3.d0)/2.d0-fea1125/2.d0+         &
          fea1225/2.d0+ fea1124*sqrt(3.d0)/2.d0)*y2*y3**2*s4b                 
               s3 = s1+ (-fea1244/2.d0-3.d0/2.d0*fea1255-                     &
          fea1245*sqrt(3.d0)/2.d0)*y2*y3*s4a**2-sqrt(3.d0)*(fea224-           &
          fea114)*y3**2*s4b/3.d0+ (fea1224*sqrt(3.d0)/2.d0+ fea1125/2.d0-     &
          fea1225/2.d0-fea1124*sqrt(3.d0)/2.d0)*y2**2*y3*s4b+                 &
          fea3355*y3**2*s4b**2+ (-fea1222-fea1112)*y2*y3**3+                  &
          fea1255*y1*y3*s4b**2+ fea1455*y1*s4a*s4b**2+                        &
          fea1244*y1*y3*s4a**2+ fea224*y2**2*s4a+ (-fea3355-fea1155/2.d0-     &
          fea1144/2.d0)*y3**2*s4a**2+ fea244*y3*s4a**2+                       &
          sqrt(3.d0)*(3.d0*fea1125+ fea1224*sqrt(3.d0)+ 3.d0*fea1225+         &
          fea1124*sqrt(3.d0))*y2*y3**2*s4a/6.d0                               
               s2 = s3-fea44*s4b**2+ fea1*y1+ fea1245*y1*y2*s4a*s4b+          &
          fea1222*y1*y2**3+ fea1234*y1*y2*y3*s4a+ fea1222*y1*y3**3+           &
          sqrt(3.d0)*(4.d0*fea3355-fea1155+                                   &
          3.d0*fea1144)*y2**2*s4a*s4b/6.d0-fea1125*y1**2*y3*s4b+              &
          fea1224*y1*y3**2*s4a+ fea1244*y1*y2*s4a**2+ fea1255*y1*y2*s4b**2-   &
          fea1*y2/2.d0-fea1*y3/2.d0+ fea1111*y1**4                            
               t4x= s2+ fea1123*y1**2*y2*y3+ fea1124*y1**2*y2*s4a+            &
          fea1224*y1*y2**2*s4a+ fea1225*y1*y2**2*s4b+                         &
          fea1124*y1**2*y3*s4a+ fea1125*y1**2*y2*s4b+ fea44*s4a**2+           &
          fea24*y2*s4a-fea11*y2**2/2.d0-fea11*y3**2/2.d0+ fea4*s4a+           &
          sqrt(3.d0)*(3.d0*fea155-fea144+ 4.d0*fea244)*y3*s4a*s4b/6.d0+       &
          fea444*s4a**3+ fea11*y1**2+ (-fea112-fea122)*y2*y3**2+ (-fea112-    &
          fea122)*y2**2*y3+ sqrt(3.d0)*(fea124*sqrt(3.d0)-                    &
          3.d0*fea135)*y2*y3*s4a/3.d0-sqrt(3.d0)*(3.d0*fea155-fea144+         &
          4.d0*fea244)*y2*s4a*s4b/6.d0+ fea112*y1**2*y2+ fea122*y1*y2**2+     &
          fea144*y1*s4a**2+ fea155*y1*s4b**2+ fea2224*y2**3*s4a-              &
          2.d0*fea1122*y2**2*y3**2-2.d0*fea12*y2*y3+ fea1112*y1**3*y3+        &
          fea112*y1**2*y3+ (-5.d0/18.d0*fea1444*sqrt(3.d0)-                   &
          fea1455*sqrt(3.d0)/6.d0+ 4.d0/9.d0*fea2444*sqrt(3.d0)-              &
          fea2445/3.d0)*y2*s4b**3                                             
   
   
          t56x = 0
          if (parmax>149) then 
             !
                  s6 = (-fea11124+ fea11135*sqrt(3.d0)/3.d0+                     &
             fea22235*sqrt(3.d0)/3.d0+ 2.d0*hea13335)*y2**3*y3*s4a+              &
             (hea233334/2.d0+ fea233334*sqrt(3.d0)/3.d0-                         &
             fea133334*sqrt(3.d0)/3.d0-fea233335/2.d0)*y1**4*y2*s4b-             &
             sqrt(3.d0)*(3.d0*fea122444*sqrt(3.d0)+ 2.d0*hea233444+              &
             2.d0*fea113444*sqrt(3.d0)-6.d0*sqrt(3.d0)*hea223555+                &
             2.d0*hea133444+ 2.d0*fea133445+                                     &
             2.d0*fea113445)*y1*y3**2*s4a*s4b**2/3.d0-                           &
             sqrt(3.d0)*(21.d0*hea234555-9.d0*hea234445+ 72.d0*fea124444-        &
             8.d0*sqrt(3.d0)*hea135555+                                          &
             48.d0*fea235555)*y1*y3*s4a**3*s4b/72.d0                             
                  s5 = s6+ (6.d0*hea33444+ 2.d0/3.d0*fea22455*sqrt(3.d0)-        &
             2.d0/3.d0*fea11455*sqrt(3.d0)+ 3.d0*fea22555)*y3**2*s4a**2*s4b-     &
             sqrt(3.d0)*(2.d0*fea111344*sqrt(3.d0)+ 3.d0*hea133344+              &
             3.d0*fea233345-3.d0*hea233344+                                      &
             fea233344*sqrt(3.d0))*y1*y3**3*s4b**2/9.d0+ (-                      &
             47.d0/72.d0*fea24555*sqrt(3.d0)+ fea15555/3.d0+                     &
             3.d0/8.d0*hea14445-25.d0/24.d0*hea34555+                            &
             5.d0/6.d0*fea24455)*y2*s4a**4-fea22555*y3**2*s4b**3+                &
             (9.d0/10.d0*hea344445-17.d0/30.d0*sqrt(3.d0)*hea344455+             &
             sqrt(3.d0)*hea345555/5.d0+ 3.d0/2.d0*hea355555+ hea144445/5.d0-     &
             3.d0/5.d0*fea145555)*y3*s4a**3*s4b**2                               
                  s4 = s5-sqrt(3.d0)*(6.d0*hea33444-fea22455*sqrt(3.d0)-         &
             2.d0*fea11455*sqrt(3.d0))*y2**2*s4a**3/9.d0-                        &
             sqrt(3.d0)*(3.d0*hea233333+                                         &
             2.d0*sqrt(3.d0)*fea111112)*y2*y3**5/3.d0-                           &
             sqrt(3.d0)*(3.d0*hea11122+                                          &
             sqrt(3.d0)*fea11122)*y2**2*y3**3/6.d0+                              &
             sqrt(3.d0)*(fea222444*sqrt(3.d0)-hea333444+                         &
             hea333455)*y3**3*s4a*s4b**2/3.d0-                                   &
             sqrt(3.d0)*(3.d0*fea113444*sqrt(3.d0)-4.d0*hea233444+               &
             4.d0*fea133445-2.d0*fea233445+ 2.d0*fea122444*sqrt(3.d0)-           &
             6.d0*sqrt(3.d0)*hea223555+                                          &
             2.d0*hea133444)*y1**2*y3*s4a*s4b**2/3.d0+ (fea45555/3.d0+           &
             2.d0/3.d0*hea55555)*s4a**5+ (-6.d0*hea33444-                        &
             2.d0/3.d0*fea22455*sqrt(3.d0)+ 2.d0/3.d0*fea11455*sqrt(3.d0)-       &
             3.d0*fea22555)*y2**2*s4a**2*s4b+ (-fea11124+                        &
             fea11135*sqrt(3.d0)/3.d0+ fea22235*sqrt(3.d0)/3.d0+                 &
             2.d0*hea13335)*y2*y3**3*s4a+ (-hea233334/2.d0-                      &
             fea233334*sqrt(3.d0)/3.d0+ fea133334*sqrt(3.d0)/3.d0+               &
             fea233335/2.d0)*y1**4*y3*s4b                                        
                  s5 = s4+ (-193.d0/300.d0*hea344445-                            &
             167.d0/300.d0*sqrt(3.d0)*hea344455-                                 &
             9.d0/50.d0*sqrt(3.d0)*hea345555+ 63.d0/20.d0*hea355555-             &
             127.d0/150.d0*hea144445-33.d0/50.d0*fea145555)*y3*s4a**5+           &
             (27.d0/4.d0*hea355555-17.d0/10.d0*fea145555-                        &
             39.d0/20.d0*hea344445-21.d0/20.d0*sqrt(3.d0)*hea344455-             &
             sqrt(3.d0)*hea345555/10.d0-21.d0/10.d0*hea144445)*y3*s4a*s4b**4+    &
             (2.d0/3.d0*fea22235*sqrt(3.d0)+ 2.d0/3.d0*fea11135*sqrt(3.d0)+      &
             hea13335)*y1*y2**3*s4a-sqrt(3.d0)*(-fea11114+                       &
             hea33335)*y3**4*s4b+ (3.d0/2.d0*hea334445+ 2.d0*hea114555-          &
             hea334555/2.d0+ 4.d0/3.d0*sqrt(3.d0)*hea334455-                     &
             6.d0*fea335555)*y3**2*s4a**2*s4b**2+ (2.d0*fea12255+                &
             2.d0*fea11255+ 3.d0*fea22344+ fea11245*sqrt(3.d0)+                  &
             fea12245*sqrt(3.d0))*y2**2*y3*s4b**2+ sqrt(3.d0)*(-fea11114+        &
             hea33335)*y2**4*s4b+ (-fea133334*sqrt(3.d0)/3.d0-hea233334+         &
             fea233334*sqrt(3.d0)/3.d0)*y1*y2**4*s4b-sqrt(3.d0)*(hea114555-      &
             hea334555)*y3**2*s4a*s4b**3/3.d0                                    
                  s3 = s5+ (-sqrt(3.d0)*hea135555/3.d0+ hea234555/8.d0+          &
             3.d0/8.d0*hea234445)*y1*y2*s4b**4+ sqrt(3.d0)*(fea233445+           &
             hea233444+ 2.d0*fea122444*sqrt(3.d0)+ 2.d0*fea113444*sqrt(3.d0)-    &
             3.d0*sqrt(3.d0)*hea223555-2.d0*hea133444+                           &
             2.d0*fea113445)*y2*y3**2*s4a*s4b**2/3.d0+ fea11112*y1**4*y3+        &
             fea11114*y1**4*s4a+ fea333334*y2**5*s4a-                            &
             sqrt(3.d0)*(fea22255*sqrt(3.d0)+ 2.d0*fea11155*sqrt(3.d0)+          &
             3.d0*fea22245)*y3**3*s4a**2/9.d0-sqrt(3.d0)*(6.d0*hea33444-         &
             fea22455*sqrt(3.d0)-2.d0*fea11455*sqrt(3.d0))*y3**2*s4a**3/9.d0+    &
             fea111122*y1**4*y3**2+ (-fea222444*sqrt(3.d0)-2.d0*hea333444+       &
             sqrt(3.d0)*hea222555-hea333455)*y2**3*s4b**3-                       &
             sqrt(3.d0)*(3.d0*hea11112+ sqrt(3.d0)*fea11112)*y2**4*y3/6.d0       
                  s6 = -sqrt(3.d0)*(3.d0*fea12255*sqrt(3.d0)+ 6.d0*fea11245+     &
             6.d0*hea11244+ 8.d0*fea22344*sqrt(3.d0)+                            &
             6.d0*fea11255*sqrt(3.d0)+ 9.d0*fea12245)*y1*y3**2*s4a**2/3.d0-      &
             sqrt(3.d0)*(3.d0*fea12255*sqrt(3.d0)+ 6.d0*fea11245+                &
             6.d0*hea11244+ 8.d0*fea22344*sqrt(3.d0)+                            &
             6.d0*fea11255*sqrt(3.d0)+ 9.d0*fea12245)*y1*y2**2*s4a**2/3.d0+ (-   &
             5.d0/8.d0*fea24555*sqrt(3.d0)+ fea15555+ 9.d0/8.d0*hea14445-        &
             9.d0/8.d0*hea34555+ 3.d0/2.d0*fea24455)*y2*s4b**4+                  &
             sqrt(3.d0)*(fea12455*sqrt(3.d0)+ 2.d0*fea23455*sqrt(3.d0)+          &
             6.d0*hea13444)*y1*y3*s4a**3/9.d0                                    
                  s5 = s6-sqrt(3.d0)*(3.d0*hea11112+                             &
             sqrt(3.d0)*fea11112)*y2*y3**4/6.d0+ (-4.d0/3.d0*fea235555-          &
             3.d0*fea124444-7.d0/9.d0*sqrt(3.d0)*hea135555-                      &
             5.d0/24.d0*hea234555-5.d0/8.d0*hea234445)*y1*y2*s4a**2*s4b**2+      &
             sqrt(3.d0)*(fea12455*sqrt(3.d0)+ 2.d0*fea23455*sqrt(3.d0)+          &
             6.d0*hea13444)*y1*y2*s4a**3/9.d0-sqrt(3.d0)*(-3.d0*fea11124+        &
             3.d0*hea13335+ fea11135*sqrt(3.d0))*y1*y3**3*s4b/3.d0+              &
             (2.d0*fea12255+ 2.d0*fea11255+ 3.d0*fea22344+                       &
             fea11245*sqrt(3.d0)+ fea12245*sqrt(3.d0))*y2*y3**2*s4b**2           
                  s6 = s5+ fea444444*s4a**6-sqrt(3.d0)*(3.d0*hea233333+          &
             2.d0*sqrt(3.d0)*fea111112)*y2**5*y3/3.d0+                           &
             sqrt(3.d0)*(2.d0*hea333355+                                         &
             sqrt(3.d0)*hea111145)*y2**4*s4b**2/6.d0+                            &
             (2.d0/3.d0*fea22235*sqrt(3.d0)+ 2.d0/3.d0*fea11135*sqrt(3.d0)+      &
             hea13335)*y1*y3**3*s4a                                              
                  s4 = s6+ sqrt(3.d0)*(-9.d0*hea14445-24.d0*fea15555+            &
             21.d0*hea34555+ 11.d0*fea24555*sqrt(3.d0)-                          &
             24.d0*fea24455)*y2*s4a**3*s4b/18.d0+ (-193.d0/300.d0*hea344445-     &
             167.d0/300.d0*sqrt(3.d0)*hea344455-                                 &
             9.d0/50.d0*sqrt(3.d0)*hea345555+ 63.d0/20.d0*hea355555-             &
             127.d0/150.d0*hea144445-33.d0/50.d0*fea145555)*y2*s4a**5+           &
             sqrt(3.d0)*(3.d0*hea233334+ 2.d0*fea133334*sqrt(3.d0)+              &
             3.d0*fea233335)*y1**4*y2*s4a/6.d0-sqrt(3.d0)*(-9.d0*hea14445-       &
             24.d0*fea15555+ 21.d0*hea34555+ 11.d0*fea24555*sqrt(3.d0)-          &
             24.d0*fea24455)*y3*s4a**3*s4b/18.d0-                                &
             sqrt(3.d0)*(3.d0*sqrt(3.d0)*hea344455+ 3.d0*hea344445+              &
             6.d0*hea144445+ 2.d0*sqrt(3.d0)*hea345555-15.d0*hea355555+          &
             6.d0*fea145555)*y2*s4a**2*s4b**3/6.d0-                              &
             sqrt(3.d0)*(3.d0*hea122333+                                         &
             sqrt(3.d0)*fea122333)*y1**3*y2*y3**2/6.d0                           
                  s5 = s4-sqrt(3.d0)*(3.d0*hea122333+                            &
             sqrt(3.d0)*fea122333)*y1**3*y2**2*y3/6.d0-                          &
             sqrt(3.d0)*(34.d0*hea144445-7.d0*hea344445+                         &
             27.d0*sqrt(3.d0)*hea344455-6.d0*sqrt(3.d0)*hea345555-               &
             45.d0*hea355555+ 18.d0*fea145555)*y2*s4a**4*s4b/60.d0+              &
             sqrt(3.d0)*(3.d0*hea122333-                                         &
             sqrt(3.d0)*fea122333)*y1**2*y2*y3**3/6.d0-sqrt(3.d0)*(-             &
             fea122334+ hea122335)*y1**2*y2**2*y3*s4b/4.d0+ sqrt(3.d0)*(-        &
             fea122334+ hea122335)*y1**2*y2*y3**2*s4b/4.d0-                      &
             sqrt(3.d0)*(4.d0*hea123344-fea112344*sqrt(3.d0)-                    &
             2.d0*sqrt(3.d0)*hea123345)*y1**2*y2*y3*s4b**2/9.d0+                 &
             2.d0/3.d0*sqrt(3.d0)*hea111333*y2**3*y3**3+                         &
             sqrt(3.d0)*(3.d0*hea11112-sqrt(3.d0)*fea11112)*y1*y2**4/6.d0+       &
             sqrt(3.d0)*(hea114555-hea334555)*y2**2*s4a*s4b**3/3.d0              
                  s2 = s5+ sqrt(3.d0)*(3.d0*hea11122-                            &
             sqrt(3.d0)*fea11122)*y1**2*y3**3/6.d0+ sqrt(3.d0)*(3.d0*hea11122-   &
             sqrt(3.d0)*fea11122)*y1**2*y2**3/6.d0-                              &
             sqrt(3.d0)*(4.d0*fea22255*sqrt(3.d0)-6.d0*fea22245-                 &
             fea11155*sqrt(3.d0))*y1**3*s4a**2/9.d0+ sqrt(3.d0)*(-               &
             55.d0*hea355555+ 22.d0*fea145555+ 27.d0*hea344445+                  &
             13.d0*sqrt(3.d0)*hea344455+ 6.d0*sqrt(3.d0)*hea345555+              &
             6.d0*hea144445)*y3*s4b**5/100.d0-                                   &
             sqrt(3.d0)*(fea22255*sqrt(3.d0)+ 2.d0*fea11155*sqrt(3.d0)+          &
             3.d0*fea22245)*y2**3*s4a**2/9.d0+ sqrt(3.d0)*(3.d0*hea11112-        &
             sqrt(3.d0)*fea11112)*y1*y3**4/6.d0-sqrt(3.d0)*(-55.d0*hea355555+    &
             22.d0*fea145555+ 27.d0*hea344445+ 13.d0*sqrt(3.d0)*hea344455+       &
             6.d0*sqrt(3.d0)*hea345555+ 6.d0*hea144445)*y2*s4b**5/100.d0+        &
             sqrt(3.d0)*(hea333335-fea333334)*y3**5*s4b/2.d0+                    &
             fea12455*y1*y2*s4a*s4b**2+ sqrt(3.d0)*(12.d0*hea33444+              &
             4.d0*fea22455*sqrt(3.d0)-fea11455*sqrt(3.d0))*y1**2*s4a**3/9.d0+    &
             s3                                                                  
                  s6 = -sqrt(3.d0)*(-9.d0*hea334445-5.d0*hea114555+              &
             2.d0*hea334555+ 24.d0*fea335555-                                    &
             4.d0*sqrt(3.d0)*hea334455)*y3**2*s4a**3*s4b/9.d0+                   &
             sqrt(3.d0)*(3.d0*fea11255*sqrt(3.d0)+ 4.d0*fea22344*sqrt(3.d0)+     &
             3.d0*fea11245+ 6.d0*fea12245+                                       &
             6.d0*hea11244)*y1**2*y3*s4a**2/3.d0+                                &
             sqrt(3.d0)*(3.d0*fea11255*sqrt(3.d0)+ 4.d0*fea22344*sqrt(3.d0)+     &
             3.d0*fea11245+ 6.d0*fea12245+                                       &
             6.d0*hea11244)*y1**2*y2*s4a**2/3.d0+ (-hea233344+                   &
             fea133344*sqrt(3.d0)/3.d0+ hea133344-                               &
             fea233344*sqrt(3.d0)/3.d0)*y1**3*y2*s4a*s4b                         
                  s5 = s6-sqrt(3.d0)*(hea333455-2.d0*fea222444*sqrt(3.d0)-       &
             7.d0*hea333444)*y1**3*s4a*s4b**2/6.d0+                              &
             sqrt(3.d0)*(3.d0*hea122333-                                         &
             sqrt(3.d0)*fea122333)*y1**2*y2**3*y3/6.d0+                          &
             sqrt(3.d0)*(hea111334+ 2.d0*hea113334+ hea223334+                   &
             fea223334*sqrt(3.d0))*y1**3*y3**2*s4a/3.d0-                         &
             sqrt(3.d0)*(3.d0*fea113444*sqrt(3.d0)-4.d0*hea233444+               &
             4.d0*fea133445-2.d0*fea233445+ 2.d0*fea122444*sqrt(3.d0)-           &
             6.d0*sqrt(3.d0)*hea223555+                                          &
             2.d0*hea133444)*y1**2*y2*s4a*s4b**2/3.d0+                           &
             sqrt(3.d0)*(2.d0*hea333355+                                         &
             sqrt(3.d0)*hea111145)*y3**4*s4b**2/6.d0                             
                  s4 = s5+ (-sqrt(3.d0)*hea111145/2.d0-fea333344*sqrt(3.d0)+     &
             hea333355)*y3**4*s4a*s4b-sqrt(3.d0)*(3.d0*hea11122+                 &
             sqrt(3.d0)*fea11122)*y2**3*y3**2/6.d0+ fea111111*y1**6-             &
             sqrt(3.d0)*(3.d0*fea122444*sqrt(3.d0)+ 2.d0*hea233444+              &
             2.d0*fea113444*sqrt(3.d0)-6.d0*sqrt(3.d0)*hea223555+                &
             2.d0*hea133444+ 2.d0*fea133445+                                     &
             2.d0*fea113445)*y1*y2**2*s4a*s4b**2/3.d0+ sqrt(3.d0)*(hea123335-    &
             fea123334)*y1*y2*y3**3*s4b/2.d0+ (-4.d0/3.d0*hea123344-             &
             sqrt(3.d0)*hea123345/3.d0-                                          &
             2.d0/3.d0*fea112344*sqrt(3.d0))*y1*y2*y3**2*s4a*s4b+                &
             sqrt(3.d0)*(34.d0*hea144445-7.d0*hea344445+                         &
             27.d0*sqrt(3.d0)*hea344455-6.d0*sqrt(3.d0)*hea345555-               &
             45.d0*hea355555+ 18.d0*fea145555)*y3*s4a**4*s4b/60.d0+ (-           &
             2.d0/3.d0*hea444445-7.d0/3.d0*fea444444)*s4a**2*s4b**4+             &
             (hea233344-fea133344*sqrt(3.d0)/3.d0-hea133344+                     &
             fea233344*sqrt(3.d0)/3.d0)*y1**3*y3*s4a*s4b+                        &
             fea133334*y1*y3**4*s4a                                              
                  s5 = s4+ fea122444*y1*y2**2*s4a**3-fea24555*y3*s4a*s4b**3+     &
             sqrt(3.d0)*(-3.d0*fea11124+ 3.d0*hea13335+                          &
             fea11135*sqrt(3.d0))*y1*y2**3*s4b/3.d0-sqrt(3.d0)*(-                &
             4.d0*fea12455*sqrt(3.d0)+ fea23455*sqrt(3.d0)+                      &
             12.d0*hea13444)*y2*y3*s4a**3/9.d0+ (fea133344*sqrt(3.d0)/3.d0-      &
             2.d0/3.d0*fea111344*sqrt(3.d0)-hea133344-fea233345-hea233344+       &
             fea233344*sqrt(3.d0)/3.d0)*y1*y3**3*s4a*s4b+                        &
             fea24455*y2*s4a**2*s4b**2+ fea122333*y1*y2**2*y3**3+                &
             fea233344*y2**3*y3*s4a**2+ (-2.d0/3.d0*hea113334+                   &
             2.d0/3.d0*hea111334-hea223334/3.d0)*y2**3*y3**2*s4b                 
                  s3 = s5+ fea133344*y1*y2**3*s4a**2+ fea133334*y1*y2**4*s4a+    &
             fea233334*y2**4*y3*s4a+ fea223334*y2**3*y3**2*s4a+                  &
             fea122333*y1*y2**3*y3**2+ fea124444*y1*y3*s4a**4-                   &
             fea233335*y2**4*y3*s4b+ fea233334*y2*y3**4*s4a+ (-                  &
             sqrt(3.d0)*hea223345/6.d0+ 2.d0/3.d0*hea113344-                     &
             fea223344*sqrt(3.d0)/3.d0)*y1**2*y2**2*s4a*s4b+                     &
             sqrt(3.d0)*(3.d0*sqrt(3.d0)*hea344455+ 3.d0*hea344445+              &
             6.d0*hea144445+ 2.d0*sqrt(3.d0)*hea345555-15.d0*hea355555+          &
             6.d0*fea145555)*y3*s4a**2*s4b**3/6.d0                               
                  s5 = (-hea444445-fea444444)*s4a**4*s4b**2+                     &
             fea15555*y1*s4b**4+ fea335555*y2**2*s4b**4+ (-                      &
             fea133344*sqrt(3.d0)/3.d0+ 2.d0/3.d0*fea111344*sqrt(3.d0)+          &
             hea133344+ fea233345+ hea233344-                                    &
             fea233344*sqrt(3.d0)/3.d0)*y1*y2**3*s4a*s4b+                        &
             (4.d0/3.d0*hea123344+ sqrt(3.d0)*hea123345/3.d0+                    &
             2.d0/3.d0*fea112344*sqrt(3.d0))*y1*y2**2*y3*s4a*s4b+                &
             fea145555*y1*s4a*s4b**4+ (27.d0/4.d0*hea355555-                     &
             17.d0/10.d0*fea145555-39.d0/20.d0*hea344445-                        &
             21.d0/20.d0*sqrt(3.d0)*hea344455-sqrt(3.d0)*hea345555/10.d0-        &
             21.d0/10.d0*hea144445)*y2*s4a*s4b**4+ fea113444*y1**2*y3*s4a**3+    &
             (9.d0/10.d0*hea344445-17.d0/30.d0*sqrt(3.d0)*hea344455+             &
             sqrt(3.d0)*hea345555/5.d0+ 3.d0/2.d0*hea355555+ hea144445/5.d0-     &
             3.d0/5.d0*fea145555)*y2*s4a**3*s4b**2                               
                  s6 = s5+ (3.d0*fea222444*sqrt(3.d0)+ 3.d0*hea333444-           &
             3.d0*sqrt(3.d0)*hea222555+ 2.d0*hea333455)*y2**3*s4a**2*s4b+        &
             fea122334*y1*y2**2*y3**2*s4a+ (-3.d0*fea222444*sqrt(3.d0)-          &
             3.d0*hea333444+ 3.d0*sqrt(3.d0)*hea222555-                          &
             2.d0*hea333455)*y3**3*s4a**2*s4b+ (fea133334*sqrt(3.d0)/3.d0+       &
             hea233334-fea233334*sqrt(3.d0)/3.d0)*y1*y3**4*s4b                   
                  s4 = s6+ (-6.d0*hea13444+ 2.d0/3.d0*fea12455*sqrt(3.d0)-       &
             2.d0/3.d0*fea23455*sqrt(3.d0)-3.d0*fea12555)*y1*y2*s4a**2*s4b+      &
             fea133344*y1*y3**3*s4a**2+ (-10.d0/3.d0*fea235555-                  &
             12.d0*fea124444-4.d0/9.d0*sqrt(3.d0)*hea135555-                     &
             10.d0/3.d0*hea234555-hea234445)*y2*y3*s4a**2*s4b**2+ (-             &
             4.d0/3.d0*fea235555-3.d0*fea124444-                                 &
             7.d0/9.d0*sqrt(3.d0)*hea135555-5.d0/24.d0*hea234555-                &
             5.d0/8.d0*hea234445)*y1*y3*s4a**2*s4b**2+ (-5.d0*fea11245-          &
             2.d0*fea12255*sqrt(3.d0)-4.d0*fea11255*sqrt(3.d0)-                  &
             6.d0*fea22344*sqrt(3.d0)-7.d0*fea12245-                             &
             6.d0*hea11244)*y2*y3**2*s4a*s4b+ (6.d0*hea13444-                    &
             2.d0/3.d0*fea12455*sqrt(3.d0)+ 2.d0/3.d0*fea23455*sqrt(3.d0)+       &
             3.d0*fea12555)*y1*y3*s4a**2*s4b                                     
                  s5 = s4-sqrt(3.d0)*(hea12335-fea11234)*y1*y2*y3**2*s4b+        &
             fea113444*y1**2*y2*s4a**3+                                          &
             sqrt(3.d0)*(2.d0*fea222444*sqrt(3.d0)+ 3.d0*hea333444+              &
             3.d0*hea333455)*y1**3*s4a**3/6.d0-                                  &
             sqrt(3.d0)*hea111333*y1**3*y3**3/3.d0+ (4.d0*fea124444+             &
             4.d0/3.d0*sqrt(3.d0)*hea135555+ hea234555+                          &
             fea235555)*y2*y3*s4a**4-sqrt(3.d0)*hea111333*y1**3*y2**3/3.d0+ (-   &
             4.d0/3.d0*fea133445+ 4.d0/3.d0*hea233444+ 2.d0/3.d0*fea233445-      &
             10.d0/9.d0*fea122444*sqrt(3.d0)-8.d0/9.d0*fea113444*sqrt(3.d0)+     &
             2.d0*sqrt(3.d0)*hea223555-2.d0*hea133444+                           &
             fea113445/3.d0)*y1**2*y2*s4b**3+ sqrt(3.d0)*(-                      &
             sqrt(3.d0)*hea111145-3.d0*fea333344*sqrt(3.d0)+                     &
             hea333355)*y1**4*s4b**2/3.d0+ (5.d0*fea11245+                       &
             2.d0*fea12255*sqrt(3.d0)+ 4.d0*fea11255*sqrt(3.d0)+                 &
             6.d0*fea22344*sqrt(3.d0)+ 7.d0*fea12245+                            &
             6.d0*hea11244)*y2**2*y3*s4a*s4b                                     
                  s1 = s5+ sqrt(3.d0)*(21.d0*hea234555-9.d0*hea234445+           &
             72.d0*fea124444-8.d0*sqrt(3.d0)*hea135555+                          &
             48.d0*fea235555)*y1*y2*s4a**3*s4b/72.d0+                            &
             fea123444*y1*y2*y3*s4a*s4b**2-                                      &
             sqrt(3.d0)*(40.d0*sqrt(3.d0)*hea135555+ 75.d0*hea234555+            &
             9.d0*hea234445+ 216.d0*fea124444+                                   &
             48.d0*fea235555)*y1*y2*s4a*s4b**3/72.d0+                            &
             fea22344*y2*y3**2*s4a**2+ fea12255*y1*y3**2*s4b**2-                 &
             fea11135*y1**3*y2*s4b+ fea11223*y1**2*y2*y3**2+                     &
             fea11224*y1**2*y2**2*s4a-fea111111*y2**6/2.d0+ s2+ s3               
                  s4 = s1+ fea22235*y2**3*y3*s4b+ fea22344*y2**2*y3*s4a**2+ (-   &
             3.d0*hea22335+ 4.d0*fea11224)*y2**2*y3**2*s4a+                      &
             fea22245*y2**3*s4a*s4b-fea22235*y2*y3**3*s4b+                       &
             fea222444*y2**3*s4a**3+ sqrt(3.d0)*(3.d0*hea233333+                 &
             sqrt(3.d0)*fea111112)*y1*y2**5/3.d0+ sqrt(3.d0)*(3.d0*hea233333+    &
             sqrt(3.d0)*fea111112)*y1*y3**5/3.d0-fea22245*y3**3*s4a*s4b+         &
             fea22455*y3**2*s4a*s4b**2+ fea12255*y1*y2**2*s4b**2-                &
             fea12555*y1*y3*s4b**3+ fea24555*y2*s4a*s4b**3+                      &
             fea233345*y2*y3**3*s4a*s4b+ fea123444*y1*y2*y3*s4a**3+              &
             fea23455*y2*y3*s4a*s4b**2+ fea123334*y1*y2*y3**3*s4a-               &
             fea233445*y2**2*y3*s4a**2*s4b                                       
                  s3 = s4-fea111111*y3**6/2.d0+ fea11223*y1**2*y2**2*y3+         &
             fea11224*y1**2*y3**2*s4a+ fea11455*y1**2*s4a*s4b**2+                &
             fea11255*y1**2*y3*s4b**2-sqrt(3.d0)*(hea22335-                      &
             fea11224)*y1**2*y2**2*s4b+ sqrt(3.d0)*(hea22335-                    &
             fea11224)*y1**2*y3**2*s4b+ sqrt(3.d0)*(hea12335-                    &
             fea11234)*y1*y2**2*y3*s4b-fea133445*y1*y2**2*s4a**2*s4b-            &
             fea11111*y2**5/2.d0+ (sqrt(3.d0)*hea223345/6.d0-                    &
             2.d0/3.d0*hea113344+                                                &
             fea223344*sqrt(3.d0)/3.d0)*y1**2*y3**2*s4a*s4b+                     &
             fea113445*y1**2*y3*s4a**2*s4b-fea113445*y1**2*y2*s4a**2*s4b+        &
             sqrt(3.d0)*(5.d0*hea123344+ 2.d0*sqrt(3.d0)*hea123345+              &
             fea112344*sqrt(3.d0))*y1*y2*y3**2*s4b**2/9.d0-                      &
             fea233345*y2**3*y3*s4a*s4b+ fea12555*y1*y2*s4b**3+                  &
             fea223344*y2**2*y3**2*s4a**2+ fea11124*y1**3*y3*s4a+                &
             fea123333*y1*y2*y3**4                                               
                  s4 = fea124444*y1*y2*s4a**4+ fea223334*y2**2*y3**3*s4a+        &
             fea11124*y1**3*y2*s4a+ fea111122*y1**4*y2**2+                       &
             fea133445*y1*y3**2*s4a**2*s4b+ (-2.d0*fea11114+                     &
             3.d0*hea33335)*y3**4*s4a+ fea112344*y1**2*y2*y3*s4a**2+             &
             fea123334*y1*y2**3*y3*s4a+ fea123333*y1*y2**4*y3+                   &
             fea122444*y1*y3**2*s4a**3+ (hea113334/3.d0+ 2.d0/3.d0*hea223334+    &
             2.d0/3.d0*hea111334)*y1**2*y3**3*s4b+ fea233344*y2*y3**3*s4a**2-    &
             fea11111*y3**5/2.d0+ (2.d0/3.d0*hea113334-2.d0/3.d0*hea111334+      &
             hea223334/3.d0)*y2**2*y3**3*s4b+ fea11123*y1**3*y2*y3+              &
             fea233335*y2*y3**4*s4b+ (fea122334/4.d0+                            &
             3.d0/4.d0*hea122335)*y1**2*y2*y3**2*s4a+ sqrt(3.d0)*(hea113334+     &
             2.d0*hea111334-hea223334+                                           &
             fea223334*sqrt(3.d0))*y1**2*y3**3*s4a/3.d0+ (fea122334/4.d0+        &
             3.d0/4.d0*hea122335)*y1**2*y2**2*y3*s4a                             
                  s5 = s4+ fea235555*y2*y3*s4b**4+ (-hea113334/3.d0-             &
             2.d0/3.d0*hea223334-2.d0/3.d0*hea111334)*y1**2*y2**3*s4b-           &
             2.d0*fea123333*y1**4*y2*y3+ (-fea123334/2.d0+                       &
             3.d0/2.d0*hea123335)*y1**3*y2*y3*s4a+ fea111344*y1**3*y3*s4a**2+    &
             fea111112*y1**5*y3+ (-2.d0/3.d0*hea113334-hea111334/3.d0+           &
             2.d0/3.d0*hea223334)*y1**3*y2**2*s4b+ fea111344*y1**3*y2*s4a**2+    &
             (2.d0/3.d0*hea113334+ hea111334/3.d0-                               &
             2.d0/3.d0*hea223334)*y1**3*y3**2*s4b                                
                  s2 = s5+ (-2.d0*fea11234+ 3.d0*hea12335)*y1*y2*y3**2*s4a+ (-   &
             hea334445-11.d0/18.d0*hea114555-hea334555/18.d0+                    &
             7.d0/3.d0*fea335555-                                                &
             13.d0/18.d0*sqrt(3.d0)*hea334455)*y1**2*s4a**4+                     &
             fea22455*y2**2*s4a*s4b**2+ fea11255*y1**2*y2*s4b**2-                &
             2.d0*fea11223*y1*y2**2*y3**2-sqrt(3.d0)*(6.d0*fea133445-            &
             3.d0*fea233445-3.d0*hea233444+ 4.d0*fea122444*sqrt(3.d0)+           &
             4.d0*fea113444*sqrt(3.d0)-9.d0*sqrt(3.d0)*hea223555+                &
             6.d0*hea133444)*y2*y3**2*s4a**3/3.d0+ (-32.d0/75.d0*hea144445-      &
             19.d0/75.d0*hea344445-11.d0/75.d0*sqrt(3.d0)*hea344455+             &
             6.d0/25.d0*sqrt(3.d0)*hea345555+ 9.d0/5.d0*hea355555-               &
             3.d0/25.d0*fea145555)*y1*s4a**5+ fea111112*y1**5*y2+                &
             fea24455*y3*s4a**2*s4b**2+ (10.d0/3.d0*hea55555-                    &
             4.d0/3.d0*fea45555)*s4a**3*s4b**2+ s3                               
                  s5 = -sqrt(3.d0)*(3.d0*hea223333+                              &
             2.d0*sqrt(3.d0)*fea111122)*y2**4*y3**2/3.d0+ (hea444445/3.d0-       &
             fea444444/3.d0)*s4b**6+ fea11111*y1**5+                             &
             fea12455*y1*y3*s4a*s4b**2+ sqrt(3.d0)*(5.d0*hea123344+              &
             2.d0*sqrt(3.d0)*hea123345+                                          &
             fea112344*sqrt(3.d0))*y1*y2**2*y3*s4b**2/9.d0-                      &
             fea12245*y1*y3**2*s4a*s4b+ fea11135*y1**3*y3*s4b+                   &
             fea12355*y1*y2*y3*s4b**2+ (-hea114555/2.d0+ hea334555/2.d0+         &
             fea335555-sqrt(3.d0)*hea334455/2.d0)*y1**2*s4b**4                   
                  s4 = s5-fea12355*y1*y2*y3*s4a**2+                              &
             fea12245*y1*y2**2*s4a*s4b+ sqrt(3.d0)*(-                            &
             2.d0*fea133344*sqrt(3.d0)+ 6.d0*hea133344+ 3.d0*fea233345-          &
             fea233344*sqrt(3.d0))*y2*y3**3*s4b**2/9.d0+ s2-                     &
             fea11245*y1**2*y3*s4a*s4b+ fea11234*y1**2*y2*y3*s4a-                &
             sqrt(3.d0)*(6.d0*fea133445-3.d0*fea233445-3.d0*hea233444+           &
             4.d0*fea122444*sqrt(3.d0)+ 4.d0*fea113444*sqrt(3.d0)-               &
             9.d0*sqrt(3.d0)*hea223555+ 6.d0*hea133444)*y2**2*y3*s4a**3/3.d0-    &
             sqrt(3.d0)*(hea123344+ sqrt(3.d0)*hea123345+                        &
             fea112344*sqrt(3.d0))*y1*y2**2*y3*s4a**2/3.d0-                      &
             sqrt(3.d0)*(3.d0*hea223333+                                         &
             2.d0*sqrt(3.d0)*fea111122)*y2**2*y3**4/3.d0-                        &
             sqrt(3.d0)*(sqrt(3.d0)*hea223345+                                   &
             2.d0*hea113344)*y1**2*y2**2*s4a**2/6.d0                             
                  s5 = s4+ (-47.d0/72.d0*fea24555*sqrt(3.d0)+ fea15555/3.d0+     &
             3.d0/8.d0*hea14445-25.d0/24.d0*hea34555+                            &
             5.d0/6.d0*fea24455)*y3*s4a**4-sqrt(3.d0)*(hea123335-                &
             fea123334)*y1*y2**3*y3*s4b/2.d0+                                    &
             sqrt(3.d0)*(fea222444*sqrt(3.d0)-hea333444+                         &
             hea333455)*y2**3*s4a*s4b**2/3.d0+ fea333334*y3**5*s4a+              &
             fea11245*y1**2*y2*s4a*s4b+ (4.d0/3.d0*fea133445-                    &
             4.d0/3.d0*hea233444-2.d0/3.d0*fea233445+                            &
             10.d0/9.d0*fea122444*sqrt(3.d0)+ 8.d0/9.d0*fea113444*sqrt(3.d0)-    &
             2.d0*sqrt(3.d0)*hea223555+ 2.d0*hea133444-                          &
             fea113445/3.d0)*y1**2*y3*s4b**3+ (-6.d0*fea15555-8.d0*fea24455+     &
             7.d0/2.d0*fea24555*sqrt(3.d0)-9.d0/2.d0*hea14445+                   &
             15.d0/2.d0*hea34555)*y1*s4a**2*s4b**2+                              &
             fea233445*y2*y3**2*s4a**2*s4b+ fea222444*y3**3*s4a**3               
                  s3 = s5+ (-8.d0/3.d0*fea24455+                                 &
             25.d0/18.d0*fea24555*sqrt(3.d0)-5.d0/3.d0*fea15555-                 &
             3.d0/2.d0*hea14445+ 11.d0/6.d0*hea34555)*y1*s4a**4+                 &
             fea22255*y3**3*s4b**2+ fea22555*y2**2*s4b**3+                       &
             fea11122*y1**3*y3**2-sqrt(3.d0)*(-sqrt(3.d0)*hea223345+             &
             hea113344+ fea223344*sqrt(3.d0))*y1**2*y2**2*s4b**2/9.d0+ (-        &
             2.d0*fea11234+ 3.d0*hea12335)*y1*y2**2*y3*s4a+                      &
             sqrt(3.d0)*(hea113334+ 2.d0*hea111334-hea223334+                    &
             fea223334*sqrt(3.d0))*y1**2*y2**3*s4a/3.d0+                         &
             sqrt(3.d0)*(3.d0*hea223333+                                         &
             sqrt(3.d0)*fea111122)*y1**2*y3**4/3.d0+                             &
             sqrt(3.d0)*(3.d0*hea223333+                                         &
             sqrt(3.d0)*fea111122)*y1**2*y2**4/3.d0+ sqrt(3.d0)*(-               &
             2.d0*fea133344*sqrt(3.d0)+ 6.d0*hea133344+ 3.d0*fea233345-          &
             fea233344*sqrt(3.d0))*y2**3*y3*s4b**2/9.d0                          
                  s6 = s3+ (hea334445/2.d0+ 2.d0/9.d0*hea114555-                 &
             7.d0/18.d0*hea334555-5.d0/3.d0*fea335555+                           &
             4.d0/9.d0*sqrt(3.d0)*hea334455)*y3**2*s4a**4+ (-                    &
             8.d0/9.d0*fea122444*sqrt(3.d0)+ 2.d0/3.d0*hea233444-                &
             10.d0/9.d0*fea113444*sqrt(3.d0)+ 2.d0*sqrt(3.d0)*hea223555-         &
             2.d0*hea133444-5.d0/3.d0*fea133445+ 2.d0/3.d0*fea113445+            &
             4.d0/3.d0*fea233445)*y1*y2**2*s4b**3-                               &
             sqrt(3.d0)*(sqrt(3.d0)*hea223345+                                   &
             2.d0*hea113344)*y1**2*y3**2*s4a**2/6.d0-sqrt(3.d0)*(-               &
             sqrt(3.d0)*hea223345+ hea113344+                                    &
             fea223344*sqrt(3.d0))*y1**2*y3**2*s4b**2/9.d0                       
                  s5 = s6+ (-fea233445/3.d0+ 2.d0/3.d0*fea133445-                &
             2.d0/3.d0*hea233444+ 2.d0/9.d0*fea122444*sqrt(3.d0)-                &
             2.d0/9.d0*fea113444*sqrt(3.d0)-                                     &
             2.d0/3.d0*fea113445)*y2**2*y3*s4b**3-fea11123*y1*y2*y3**3/2.d0+     &
             (8.d0/9.d0*fea122444*sqrt(3.d0)-2.d0/3.d0*hea233444+                &
             10.d0/9.d0*fea113444*sqrt(3.d0)-2.d0*sqrt(3.d0)*hea223555+          &
             2.d0*hea133444+ 5.d0/3.d0*fea133445-2.d0/3.d0*fea113445-            &
             4.d0/3.d0*fea233445)*y1*y3**2*s4b**3-fea11123*y1*y2**3*y3/2.d0+     &
             (fea222444*sqrt(3.d0)+ 2.d0*hea333444-sqrt(3.d0)*hea222555+         &
             hea333455)*y3**3*s4b**3                                             
                  s4 = s5-sqrt(3.d0)*(hea333335-fea333334)*y2**5*s4b/2.d0-       &
             sqrt(3.d0)*(hea123344+ sqrt(3.d0)*hea123345+                        &
             fea112344*sqrt(3.d0))*y1*y2*y3**2*s4a**2/3.d0+                      &
             fea333344*y2**4*s4a**2+ (hea334445/2.d0+ 2.d0/9.d0*hea114555-       &
             7.d0/18.d0*hea334555-5.d0/3.d0*fea335555+                           &
             4.d0/9.d0*sqrt(3.d0)*hea334455)*y2**2*s4a**4+                       &
             fea11155*y1**3*s4b**2+ (3.d0/2.d0*hea333335-                        &
             fea333334/2.d0)*y1**5*s4a+ fea11112*y1**4*y2-                       &
             sqrt(3.d0)*(2.d0*fea111344*sqrt(3.d0)+ 3.d0*hea133344+              &
             3.d0*fea233345-3.d0*hea233344+                                      &
             fea233344*sqrt(3.d0))*y1*y2**3*s4b**2/9.d0+ (-                      &
             5.d0/8.d0*fea24555*sqrt(3.d0)+ fea15555+ 9.d0/8.d0*hea14445-        &
             9.d0/8.d0*hea34555+ 3.d0/2.d0*fea24455)*y3*s4b**4+                  &
             fea45555*s4a*s4b**4                                                 
                  s5 = s4+ fea22255*y2**3*s4b**2+ (-2.d0*fea11114+               &
             3.d0*hea33335)*y2**4*s4a+ fea335555*y3**2*s4b**4+                   &
             fea333344*y3**4*s4a**2+ sqrt(3.d0)*(fea233445+ hea233444+           &
             2.d0*fea122444*sqrt(3.d0)+ 2.d0*fea113444*sqrt(3.d0)-               &
             3.d0*sqrt(3.d0)*hea223555-2.d0*hea133444+                           &
             2.d0*fea113445)*y2**2*y3*s4a*s4b**2/3.d0+                           &
             sqrt(3.d0)*(8.d0*hea113344+ sqrt(3.d0)*hea223345-                   &
             fea223344*sqrt(3.d0))*y2**2*y3**2*s4b**2/9.d0+ sqrt(3.d0)*(-        &
             9.d0*hea334445-5.d0*hea114555+ 2.d0*hea334555+ 24.d0*fea335555-     &
             4.d0*sqrt(3.d0)*hea334455)*y2**2*s4a**3*s4b/9.d0+                   &
             fea11122*y1**3*y2**2-sqrt(3.d0)*(3.d0*hea233344+                    &
             fea133344*sqrt(3.d0)+ 3.d0*hea133344+ fea233344*sqrt(3.d0)+         &
             fea111344*sqrt(3.d0))*y1**3*y2*s4b**2/9.d0                          
                  s6 = s5+ (sqrt(3.d0)*hea111145/2.d0+ fea333344*sqrt(3.d0)-     &
             hea333355)*y2**4*s4a*s4b+ (-3.d0*hea334445-2.d0*hea114555+          &
             2.d0*hea334555+ 6.d0*fea335555-                                     &
             5.d0/3.d0*sqrt(3.d0)*hea334455)*y1**2*s4a**2*s4b**2+                &
             (fea233445/3.d0-2.d0/3.d0*fea133445+ 2.d0/3.d0*hea233444-           &
             2.d0/9.d0*fea122444*sqrt(3.d0)+ 2.d0/9.d0*fea113444*sqrt(3.d0)+     &
             2.d0/3.d0*fea113445)*y2*y3**2*s4b**3+ (6.d0/5.d0*fea145555+         &
             8.d0/5.d0*hea144445+ 11.d0/5.d0*hea344445+                          &
             17.d0/15.d0*sqrt(3.d0)*hea344455-2.d0/5.d0*sqrt(3.d0)*hea345555-    &
             3.d0*hea355555)*y1*s4a**3*s4b**2-sqrt(3.d0)*(3.d0*hea233344+        &
             fea133344*sqrt(3.d0)+ 3.d0*hea133344+ fea233344*sqrt(3.d0)+         &
             fea111344*sqrt(3.d0))*y1**3*y3*s4b**2/9.d0                          
                  t56x=s6+ sqrt(3.d0)*(40.d0*sqrt(3.d0)*hea135555+               &
             75.d0*hea234555+ 9.d0*hea234445+ 216.d0*fea124444+                  &
             48.d0*fea235555)*y1*y3*s4a*s4b**3/72.d0+ sqrt(3.d0)*(hea111334+     &
             2.d0*hea113334+ hea223334+                                          &
             fea223334*sqrt(3.d0))*y1**3*y2**2*s4a/3.d0+ (-                      &
             sqrt(3.d0)*hea135555/3.d0+ hea234555/8.d0+                          &
             3.d0/8.d0*hea234445)*y1*y3*s4b**4-sqrt(3.d0)*(3.d0*hea333355-       &
             fea333344*sqrt(3.d0))*y1**4*s4a**2/3.d0+ (3.d0/2.d0*hea334445+      &
             2.d0*hea114555-hea334555/2.d0+ 4.d0/3.d0*sqrt(3.d0)*hea334455-      &
             6.d0*fea335555)*y2**2*s4a**2*s4b**2+ sqrt(3.d0)*(3.d0*hea233334+    &
             2.d0*fea133334*sqrt(3.d0)+ 3.d0*fea233335)*y1**4*y3*s4a/6.d0        
           !
           endif                                                                      
           !                                                                
           dipol_xy =( t4x+ t56x )                                        
           !                                                                   
       case (2)                                                       
                                                                              
               s3 = sqrt(3.d0)*fea1*y2/2.d0+ (2.d0/3.d0*fea1114+              &
          fea2224/3.d0)*y2**3*s4b+ (fea24/3.d0+ 2.d0/3.d0*fea14)*y3*s4b+ (-   &
          fea1114/3.d0+ 4.d0/3.d0*fea2224)*y1**3*s4b+ (fea224/3.d0+           &
          2.d0/3.d0*fea114)*y2**2*s4b+ (2.d0/3.d0*fea1114+                    &
          fea2224/3.d0)*y3**3*s4b+ (-2.d0*fea4444-fea4455)*s4a**3*s4b-        &
          sqrt(3.d0)*(-fea112+ fea122)*y2**2*y3/3.d0-fea135*y1*y2*s4a+        &
          sqrt(3.d0)*(fea122+ 2.d0*fea112)*y1*y2**2/3.d0-                     &
          sqrt(3.d0)*(fea112+ 2.d0*fea122)*y1**2*y3/3.d0+ sqrt(3.d0)*(-       &
          fea112+ fea122)*y2*y3**2/3.d0                                       
               s2 = s3-sqrt(3.d0)*fea111*y3**3/2.d0-                          &
          sqrt(3.d0)*fea1111*y3**4/2.d0+ sqrt(3.d0)*fea111*y2**3/2.d0+        &
          sqrt(3.d0)*(fea3355-fea1155)*y3**2*s4a**2/3.d0-sqrt(3.d0)*(-        &
          fea1114+ fea2224)*y3**3*s4a/3.d0+                                   &
          sqrt(3.d0)*(16.d0*fea2444*sqrt(3.d0)+ 17.d0*fea1444*sqrt(3.d0)+     &
          3.d0*fea1455*sqrt(3.d0)-12.d0*fea2445)*y2*s4b**3/108.d0+            &
          sqrt(3.d0)*(-fea1444-3.d0*fea1455+                                  &
          4.d0*fea2444)*y2*s4a**3/12.d0+ sqrt(3.d0)*(2.d0*fea3355+            &
          fea1155+ 3.d0*fea1144)*y2**2*s4b**2/6.d0+ fea135*y1*y3*s4a-         &
          sqrt(3.d0)*(fea3355-fea1155)*y2**2*s4a**2/3.d0+ sqrt(3.d0)*(-       &
          fea1114+ fea2224)*y2**3*s4a/3.d0+ sqrt(3.d0)*(fea224-               &
          fea114)*y2**2*s4a/3.d0                                              
               s3 = -sqrt(3.d0)*(fea1222-fea1112)*y2**3*y3/3.d0+              &
          sqrt(3.d0)*(20.d0*fea2444*sqrt(3.d0)+ fea1444*sqrt(3.d0)-           &
          3.d0*fea1455*sqrt(3.d0)+ 12.d0*fea2445)*y1*s4b**3/54.d0+            &
          sqrt(3.d0)*(fea1222+ 2.d0*fea1112)*y1*y2**3/3.d0-                   &
          sqrt(3.d0)*fea1122*y1**2*y3**2+ sqrt(3.d0)*(2.d0*fea1222+           &
          fea1112)*y1**3*y2/3.d0+ sqrt(3.d0)*(fea112+                         &
          2.d0*fea122)*y1**2*y2/3.d0+ (-fea155-fea144/3.d0-                   &
          8.d0/3.d0*fea244)*y1*s4a*s4b-sqrt(3.d0)*(2.d0*fea3355+ fea1155+     &
          3.d0*fea1144)*y3**2*s4b**2/6.d0+ (2.d0/3.d0*fea3355+                &
          5.d0/6.d0*fea1155-fea1144/2.d0)*y3**2*s4a*s4b-sqrt(3.d0)*(fea224-   &
          fea114)*y3**2*s4a/3.d0+ (2.d0/3.d0*fea3355+ 5.d0/6.d0*fea1155-      &
          fea1144/2.d0)*y2**2*s4a*s4b-sqrt(3.d0)*(-fea144+                    &
          fea244)*y2*s4b**2/3.d0                                              
               s1 = s3+ fea444*s4a**2*s4b+ (-fea1224*sqrt(3.d0)/6.d0+         &
          fea1125/2.d0-fea1225/2.d0+ fea1124*sqrt(3.d0)/6.d0)*y2*y3**2*s4a-   &
          sqrt(3.d0)*(fea24-fea14)*y3*s4a/3.d0+ sqrt(3.d0)*(fea24-            &
          fea14)*y2*s4a/3.d0+ sqrt(3.d0)*(16.d0*fea2444*sqrt(3.d0)+           &
          17.d0*fea1444*sqrt(3.d0)+ 3.d0*fea1455*sqrt(3.d0)-                  &
          12.d0*fea2445)*y3*s4b**3/108.d0-sqrt(3.d0)*(fea122+                 &
          2.d0*fea112)*y1*y3**2/3.d0+ (-2.d0*fea4444+                         &
          fea4455/3.d0)*s4a*s4b**3+ (fea24/3.d0+ 2.d0/3.d0*fea14)*y2*s4b+     &
          sqrt(3.d0)*fea1122*y1**2*y2**2+ (fea224/3.d0+                       &
          2.d0/3.d0*fea114)*y3**2*s4b+ sqrt(3.d0)*(3.d0*fea155+ fea144+       &
          2.d0*fea244)*y2*s4a**2/6.d0+ (4.d0/3.d0*fea224-                     &
          fea114/3.d0)*y1**2*s4b+ s2                                          
               s3 = s1+ (fea1224*sqrt(3.d0)/6.d0-fea1125/2.d0+                &
          fea1225/2.d0-fea1124*sqrt(3.d0)/6.d0)*y2**2*y3*s4a-                 &
          sqrt(3.d0)*fea12*y1*y3-sqrt(3.d0)*(3.d0*fea155+ fea144+             &
          2.d0*fea244)*y3*s4a**2/6.d0+ (fea1124*sqrt(3.d0)/3.d0-              &
          fea1224*sqrt(3.d0)/3.d0+ fea1225)*y1**2*y2*s4a+ (-                  &
          fea1124*sqrt(3.d0)/3.d0+ fea1224*sqrt(3.d0)/3.d0-                   &
          fea1225)*y1**2*y3*s4a-sqrt(3.d0)*(2.d0*fea1222+                     &
          fea1112)*y1**3*y3/3.d0+ (8.d0/3.d0*fea3355+ fea1155/3.d0+           &
          fea1144)*y1**2*s4a*s4b+ sqrt(3.d0)*(fea1222-                        &
          fea1112)*y2*y3**3/3.d0-sqrt(3.d0)*(-fea1444-3.d0*fea1455+           &
          4.d0*fea2444)*y3*s4a**3/12.d0-sqrt(3.d0)*(fea1222+                  &
          2.d0*fea1112)*y1*y3**3/3.d0+ (fea155/2.d0-5.d0/6.d0*fea144-         &
          2.d0/3.d0*fea244)*y2*s4a*s4b                                        
               s2 = s3+ (fea155/2.d0-5.d0/6.d0*fea144-                        &
          2.d0/3.d0*fea244)*y3*s4a*s4b+ sqrt(3.d0)*(-fea144+                  &
          fea244)*y3*s4b**2/3.d0+ sqrt(3.d0)*fea12*y1*y2+                     &
          sqrt(3.d0)*fea1111*y2**4/2.d0+ (4.d0/3.d0*fea24-                    &
          fea14/3.d0)*y1*s4b-sqrt(3.d0)*fea1*y3/2.d0-2.d0*fea44*s4a*s4b-      &
          sqrt(3.d0)*fea11*y3**2/2.d0+ sqrt(3.d0)*fea11*y2**2/2.d0-           &
          sqrt(3.d0)*(-fea124*sqrt(3.d0)+ 2.d0*fea135)*y1*y3*s4b/3.d0-        &
          sqrt(3.d0)*(3.d0*fea1244+ 3.d0*fea1255+                             &
          fea1245*sqrt(3.d0))*y1*y3*s4b**2/6.d0-sqrt(3.d0)*(-                 &
          fea124*sqrt(3.d0)+ 2.d0*fea135)*y1*y2*s4b/3.d0                      
               s4 = s2+ sqrt(3.d0)*(4.d0*fea2444*sqrt(3.d0)-                  &
          fea1444*sqrt(3.d0)-fea1455*sqrt(3.d0)-                              &
          4.d0*fea2445)*y1*s4a**2*s4b/6.d0+ sqrt(3.d0)*(-fea1125+             &
          fea1124*sqrt(3.d0)-fea1225+                                         &
          fea1224*sqrt(3.d0))*y2**2*y3*s4b/6.d0+ sqrt(3.d0)*(-3.d0*fea1244-   &
          3.d0*fea1255+ fea1245*sqrt(3.d0))*y1*y3*s4a**2/6.d0+                &
          sqrt(3.d0)*(fea1125+ fea1124*sqrt(3.d0)+                            &
          fea1225)*y1*y3**2*s4b/3.d0-sqrt(3.d0)*(-fea1124+                    &
          fea1125*sqrt(3.d0)+ fea1224)*y1*y3**2*s4a/3.d0                      
               s3 = s4+ sqrt(3.d0)*(fea1125+ fea1124*sqrt(3.d0)+              &
          fea1225)*y1*y2**2*s4b/3.d0+ (-fea1245*sqrt(3.d0)/3.d0-fea1244+      &
          fea1255)*y2*y3*s4a*s4b+ sqrt(3.d0)*(-fea1124+                       &
          fea1125*sqrt(3.d0)+ fea1224)*y1*y2**2*s4a/3.d0-sqrt(3.d0)*(-        &
          3.d0*fea1244-3.d0*fea1255+ fea1245*sqrt(3.d0))*y1*y2*s4a**2/6.d0-   &
          sqrt(3.d0)*fea1123*y1*y2*y3**2/2.d0+ sqrt(3.d0)*(3.d0*fea1244+      &
          3.d0*fea1255+ fea1245*sqrt(3.d0))*y1*y2*s4b**2/6.d0-                &
          sqrt(3.d0)*(4.d0*fea2444-7.d0*fea1444+                              &
          3.d0*fea1455)*y3*s4a*s4b**2/12.d0                                   
               t4y= s3+ sqrt(3.d0)*fea1123*y1*y2**2*y3/2.d0+                  &
          sqrt(3.d0)*(3.d0*fea1444*sqrt(3.d0)+ fea1455*sqrt(3.d0)+            &
          4.d0*fea2445)*y3*s4a**2*s4b/12.d0+ sqrt(3.d0)*(-fea1125+            &
          fea1124*sqrt(3.d0)-fea1225+                                         &
          fea1224*sqrt(3.d0))*y2*y3**2*s4b/6.d0+                              &
          (2.d0/3.d0*fea1245*sqrt(3.d0)-fea1244+ fea1255)*y1*y3*s4a*s4b+      &
          sqrt(3.d0)*(4.d0*fea2444-7.d0*fea1444+                              &
          3.d0*fea1455)*y2*s4a*s4b**2/12.d0+                                  &
          sqrt(3.d0)*(3.d0*fea1444*sqrt(3.d0)+ fea1455*sqrt(3.d0)+            &
          4.d0*fea2445)*y2*s4a**2*s4b/12.d0+ sqrt(3.d0)*(fea1125+             &
          fea1224*sqrt(3.d0)+ fea1225)*y1**2*y2*s4b/3.d0+                     &
          sqrt(3.d0)*(fea1125+ fea1224*sqrt(3.d0)+                            &
          fea1225)*y1**2*y3*s4b/3.d0+ fea4*s4b+                               &
          (2.d0/3.d0*fea1245*sqrt(3.d0)-fea1244+ fea1255)*y1*y2*s4a*s4b+      &
          fea1234*y1*y2*y3*s4b+ sqrt(3.d0)*(fea135+                           &
          fea124*sqrt(3.d0))*y2*y3*s4b/3.d0+ fea444*s4b**3                    
          !                                                                 
          !                                                                  
          t56y = 0
          if (parmax>149) then 
            !                                                                     
                 s5 = hea333335*y3**5*s4b+ (2.d0/9.d0*fea111344*sqrt(3.d0)-     &
            hea233344/3.d0-fea233344*sqrt(3.d0)/3.d0+                           &
            fea133344*sqrt(3.d0)/9.d0-fea233345/3.d0)*y1*y2**3*s4b**2+          &
            (17.d0/5.d0*hea355555-24.d0/25.d0*fea145555-9.d0/25.d0*hea344445-   &
            21.d0/25.d0*sqrt(3.d0)*hea344455-2.d0/25.d0*sqrt(3.d0)*hea345555-   &
            27.d0/25.d0*hea144445)*y1*s4b**5+ (-hea233334/2.d0-                 &
            fea133334*sqrt(3.d0)/3.d0+ fea233334*sqrt(3.d0)/3.d0+               &
            fea233335/2.d0)*y1**4*y2*s4a+ hea333355*y3**4*s4b**2+               &
            (7.d0*hea355555-9.d0/5.d0*hea344445-12.d0/5.d0*hea144445-           &
            6.d0/5.d0*sqrt(3.d0)*hea344455+ 4.d0/15.d0*sqrt(3.d0)*hea345555-    &
            4.d0/5.d0*fea145555)*y2*s4a**2*s4b**3-hea333355*y2**4*s4b**2+       &
            hea233333*y2*y3**5                                                  
                 s4 = s5+ sqrt(3.d0)*fea11111*y2**5/2.d0+                       &
            (sqrt(3.d0)*fea11122/2.d0-hea11122/2.d0)*y2**3*y3**2-               &
            hea11122*y1**3*y3**2-hea111333*y1**3*y2**3+ hea355555*y3*s4b**5+    &
            (2.d0/3.d0*hea444445-8.d0/3.d0*fea444444)*s4a**3*s4b**3+            &
            sqrt(3.d0)*fea111111*y2**6/2.d0+ hea444445*s4a**5*s4b+ (-           &
            hea444445/3.d0-8.d0/3.d0*fea444444)*s4a*s4b**5                      
                 s3 = s4+ hea333444*y3**3*s4a**3+ (-2.d0*hea334445-             &
            5.d0/3.d0*hea114555+ 2.d0/3.d0*hea334555+ 8.d0*fea335555-           &
            4.d0/3.d0*sqrt(3.d0)*hea334455)*y1**2*s4a**3*s4b+ (-                &
            4.d0/9.d0*fea133344*sqrt(3.d0)+ 4.d0/9.d0*fea111344*sqrt(3.d0)+     &
            fea233345/3.d0+ hea233344/3.d0)*y2**3*y3*s4b**2+ (-                 &
            2.d0/3.d0*fea12255*sqrt(3.d0)-2.d0*fea11245-3.d0*hea11244-          &
            3.d0*fea22344*sqrt(3.d0)-7.d0/3.d0*fea11255*sqrt(3.d0)-             &
            4.d0*fea12245)*y2*y3**2*s4b**2+ hea222555*y3**3*s4b**3-             &
            hea11112*y1**4*y3+ hea333335*y2**5*s4b+ hea11112*y1**4*y2+          &
            (hea223333+ sqrt(3.d0)*fea111122)*y1**2*y2**4+                      &
            hea11122*y1**3*y2**2+ hea222555*y2**3*s4b**3-                       &
            hea135555*y1*y2*s4b**4-hea133444*y1*y2**2*s4a**3+                   &
            hea334445*y3**2*s4a**3*s4b+ (-hea333335/2.d0+                       &
            3.d0/2.d0*fea333334)*y1**5*s4b-hea233444*y2**2*y3*s4a**3+           &
            (sqrt(3.d0)*fea11122/2.d0+ hea11122/2.d0)*y1**2*y2**3+              &
            (hea11112/2.d0-sqrt(3.d0)*fea11112/2.d0)*y2*y3**4                   
                 s4 = s3+ (-hea11112/2.d0+                                      &
            sqrt(3.d0)*fea11112/2.d0)*y2**4*y3-hea344455*y2*s4a**3*s4b**2-      &
            hea345555*y2*s4a*s4b**4-sqrt(3.d0)*fea111111*y3**6/2.d0+            &
            hea334445*y2**2*s4a**3*s4b-hea233333*y2**5*y3+                      &
            hea223334*y2**2*y3**3*s4a-hea113344*y1**2*y2**2*s4a**2+             &
            hea233334*y2*y3**4*s4a-hea233334*y2**4*y3*s4a-                      &
            hea333455*y2**3*s4a*s4b**2+ hea223555*y2**2*y3*s4b**3+              &
            hea334555*y2**2*s4a*s4b**3-hea334455*y2**2*s4a**2*s4b**2-           &
            hea113334*y1**2*y2**3*s4a-hea223334*y2**3*y3**2*s4a+                &
            hea233444*y2*y3**2*s4a**3                                           
                 s5 = s4+ (4.d0*hea33335-3.d0*fea11114)*y1**4*s4b+              &
            (fea233445+ hea233444-hea133444-fea133445+                          &
            fea113445)*y1**2*y2*s4a**3-hea122333*y1*y2**3*y3**2-                &
            hea133344*y1*y2**3*s4a**2-hea111334*y1**3*y2**2*s4a+                &
            2.d0*fea12355*y1*y2*y3*s4a*s4b+ hea334455*y3**2*s4a**2*s4b**2+      &
            hea22335*y2**2*y3**2*s4b+ (-hea223333-                              &
            sqrt(3.d0)*fea111122)*y1**2*y3**4                                   
                 s6 = s5-sqrt(3.d0)*(2.d0*fea133344*sqrt(3.d0)+                 &
            4.d0*fea111344*sqrt(3.d0)+ 6.d0*hea133344+                          &
            3.d0*fea233345)*y2**3*y3*s4a*s4b/9.d0+ (fea22255*sqrt(3.d0)/3.d0-   &
            fea11155*sqrt(3.d0)/3.d0-fea22245)*y3**3*s4b**2+                    &
            sqrt(3.d0)*(3.d0*hea233334+ fea133334*sqrt(3.d0)+                   &
            2.d0*fea233334*sqrt(3.d0)+ 3.d0*fea233335)*y1*y2**4*s4b/9.d0-       &
            hea33444*y2**2*s4a**3                                               
                 s2 = s6+ 2.d0/3.d0*sqrt(3.d0)*(2.d0*hea123344+                 &
            sqrt(3.d0)*hea123345+ fea112344*sqrt(3.d0))*y1**2*y2*y3*s4a*s4b-    &
            sqrt(3.d0)*(-3.d0*fea233445-3.d0*hea233444+                         &
            4.d0*fea122444*sqrt(3.d0)+ 5.d0*fea113444*sqrt(3.d0)-               &
            12.d0*sqrt(3.d0)*hea223555+ 9.d0*hea133444+                         &
            9.d0*fea133445)*y1**2*y2*s4b**3/9.d0+ (-hea11112/2.d0-              &
            sqrt(3.d0)*fea11112/2.d0)*y1*y3**4+ fea123444*y1*y2*y3*s4b**3+      &
            sqrt(3.d0)*(4.d0*fea133334*sqrt(3.d0)+ 3.d0*hea233334-              &
            fea233334*sqrt(3.d0)+ 3.d0*fea233335)*y2*y3**4*s4b/9.d0+ (-         &
            2.d0/3.d0*fea233445-hea233444-2.d0/3.d0*fea122444*sqrt(3.d0)+       &
            2.d0/3.d0*fea113444*sqrt(3.d0)+ 2.d0/3.d0*fea133445-                &
            2.d0/3.d0*fea113445)*y2*y3**2*s4a*s4b**2                            
                 s5 = s2+ (-sqrt(3.d0)*fea11122/2.d0-                           &
            hea11122/2.d0)*y1**2*y3**3-                                         &
            sqrt(3.d0)*(56.d0*sqrt(3.d0)*hea135555+ 69.d0*hea234555-            &
            9.d0*hea234445+ 216.d0*fea124444+                                   &
            96.d0*fea235555)*y1*y2*s4a**2*s4b**2/72.d0+ sqrt(3.d0)*(-           &
            2.d0*fea111344*sqrt(3.d0)-3.d0*hea233344-                           &
            3.d0*fea233344*sqrt(3.d0)-fea133344*sqrt(3.d0)+ 3.d0*hea133344+     &
            3.d0*fea233345)*y1*y2**3*s4a*s4b/9.d0+ sqrt(3.d0)*(-                &
            3.d0*sqrt(3.d0)*hea222555+ 5.d0*hea333444+                          &
            4.d0*fea222444*sqrt(3.d0)+                                          &
            3.d0*hea333455)*y2**3*s4a**2*s4b/3.d0+ (-                           &
            fea22255*sqrt(3.d0)/3.d0+ fea11155*sqrt(3.d0)/3.d0+                 &
            fea22245)*y2**3*s4b**2+ sqrt(3.d0)*(-5.d0*fea133445+                &
            4.d0*hea233444+ 2.d0*fea233445-3.d0*fea122444*sqrt(3.d0)-           &
            2.d0*fea113444*sqrt(3.d0)+ 6.d0*sqrt(3.d0)*hea223555-               &
            5.d0*hea133444+ 2.d0*fea113445)*y1*y2**2*s4a**2*s4b/3.d0+ (-        &
            2.d0*hea233333-sqrt(3.d0)*fea111112)*y1**5*y3+                      &
            sqrt(3.d0)*(fea223334*sqrt(3.d0)+ 2.d0*hea113334+                   &
            2.d0*hea111334)*y2**2*y3**3*s4b/3.d0                                
                 s4 = s5+ (5.d0/3.d0*hea55555-                                  &
            2.d0/3.d0*fea45555)*s4a**4*s4b+ (hea11112/2.d0+                     &
            sqrt(3.d0)*fea11112/2.d0)*y1*y2**4-sqrt(3.d0)*(-6.d0*fea233445-     &
            6.d0*hea233444+ 5.d0*fea122444*sqrt(3.d0)+                          &
            4.d0*fea113444*sqrt(3.d0)-12.d0*sqrt(3.d0)*hea223555+               &
            9.d0*hea133444+ 9.d0*fea133445)*y1*y3**2*s4b**3/9.d0+ (-            &
            sqrt(3.d0)*fea111112-hea233333)*y1*y3**5+ hea33444*y3**2*s4a**3+    &
            (2.d0*hea233333+ sqrt(3.d0)*fea111112)*y1**5*y2+ (-                 &
            sqrt(3.d0)*fea11122/2.d0+ hea11122/2.d0)*y2**2*y3**3+               &
            hea33335*y3**4*s4b+ hea111333*y1**3*y3**3                           
                 s5 = s4+ hea33335*y2**4*s4b-hea123344*y1*y2**2*y3*s4a**2+      &
            sqrt(3.d0)*(4.d0*fea133334*sqrt(3.d0)+ 3.d0*hea233334-              &
            fea233334*sqrt(3.d0)+ 3.d0*fea233335)*y2**4*y3*s4b/9.d0-            &
            sqrt(3.d0)*(2.d0*fea233445+ 2.d0*hea233444-                         &
            2.d0*fea122444*sqrt(3.d0)-2.d0*fea113444*sqrt(3.d0)+                &
            3.d0*sqrt(3.d0)*hea223555-4.d0*hea133444-2.d0*fea133445+            &
            2.d0*fea113445)*y2**2*y3*s4a**2*s4b/3.d0+ (-2.d0*hea223333-         &
            sqrt(3.d0)*fea111122)*y1**4*y3**2-sqrt(3.d0)*(2.d0*fea233445+       &
            2.d0*hea233444-2.d0*fea122444*sqrt(3.d0)-                           &
            2.d0*fea113444*sqrt(3.d0)+ 3.d0*sqrt(3.d0)*hea223555-               &
            4.d0*hea133444-2.d0*fea133445+                                      &
            2.d0*fea113445)*y2*y3**2*s4a**2*s4b/3.d0+                           &
            sqrt(3.d0)*(5.d0*hea34555-6.d0*fea24455+                            &
            3.d0*fea24555*sqrt(3.d0)-4.d0*fea15555-                             &
            3.d0*hea14445)*y2*s4b**4/4.d0+ (sqrt(3.d0)*fea111112+               &
            hea233333)*y1*y2**5                                                 
                 s3 = s5+ sqrt(3.d0)*(fea223334*sqrt(3.d0)+ 2.d0*hea113334+     &
            2.d0*hea111334)*y2**3*y3**2*s4b/3.d0+                               &
            hea123344*y1*y2*y3**2*s4a**2+ hea123345*y1*y2*y3**2*s4a*s4b+        &
            sqrt(3.d0)*(sqrt(3.d0)*hea223345+ 4.d0*hea113344-                   &
            2.d0*fea223344*sqrt(3.d0))*y1**2*y3**2*s4a*s4b/6.d0+                &
            sqrt(3.d0)*(3.d0*hea33444+ fea22455*sqrt(3.d0)+                     &
            fea22555)*y3**2*s4b**3/3.d0+ hea355555*y2*s4b**5-                   &
            hea223333*y2**4*y3**2+ hea223333*y2**2*y3**4-                       &
            sqrt(3.d0)*fea11223*y1**2*y2*y3**2+                                 &
            fea123444*y1*y2*y3*s4a**2*s4b                                       
                 s5 = s3+ hea123335*y1*y2*y3**3*s4b-sqrt(3.d0)*(hea123335-      &
            fea123334)*y1*y2**3*y3*s4a/2.d0+ sqrt(3.d0)*(9.d0*hea13444+         &
            2.d0*fea23455*sqrt(3.d0)+ fea12455*sqrt(3.d0)+                      &
            9.d0*fea12555)*y1*y2*s4a**2*s4b/9.d0+ sqrt(3.d0)*(-fea122334+       &
            hea122335)*y1**2*y2*y3**2*s4a/4.d0-                                 &
            sqrt(3.d0)*(10.d0*fea222444*sqrt(3.d0)+ 17.d0*hea333444-            &
            12.d0*sqrt(3.d0)*hea222555+                                         &
            9.d0*hea333455)*y1**3*s4a**2*s4b/6.d0-                              &
            sqrt(3.d0)*fea11111*y3**5/2.d0+ 2.d0*fea45555*s4a**2*s4b**3+        &
            (3.d0/4.d0*fea122334+ hea122335/4.d0)*y1**2*y2*y3**2*s4b            
                 s6 = s5-sqrt(3.d0)*(3.d0*fea133445-hea233444-                  &
            3.d0*fea233445+ 3.d0*fea113444*sqrt(3.d0)-                          &
            6.d0*sqrt(3.d0)*hea223555+ 5.d0*hea133444+                          &
            2.d0*fea122444*sqrt(3.d0))*y1**2*y2*s4a**2*s4b/3.d0-                &
            hea333444*y2**3*s4a**3+ sqrt(3.d0)*(-5.d0*fea133445+                &
            4.d0*hea233444+ 2.d0*fea233445-3.d0*fea122444*sqrt(3.d0)-           &
            2.d0*fea113444*sqrt(3.d0)+ 6.d0*sqrt(3.d0)*hea223555-               &
            5.d0*hea133444+ 2.d0*fea113445)*y1*y3**2*s4a**2*s4b/3.d0-           &
            sqrt(3.d0)*(-fea22455*sqrt(3.d0)+ 9.d0*hea33444-                    &
            2.d0*fea11455*sqrt(3.d0)+ 9.d0*fea22555)*y3**2*s4a**2*s4b/9.d0      
                 s4 = s6+ sqrt(3.d0)*(4.d0*fea22255*sqrt(3.d0)-                 &
            3.d0*fea22245+ 2.d0*fea11155*sqrt(3.d0))*y3**3*s4a*s4b/9.d0+        &
            sqrt(3.d0)*(hea113334+ hea223334+                                   &
            fea223334*sqrt(3.d0))*y1**2*y2**3*s4b/3.d0+                         &
            sqrt(3.d0)*(6.d0*hea333355-sqrt(3.d0)*hea111145-                    &
            6.d0*fea333344*sqrt(3.d0))*y2**4*s4a*s4b/6.d0+ (-3.d0*hea13444-     &
            2.d0/3.d0*fea23455*sqrt(3.d0)+                                      &
            2.d0/3.d0*fea12455*sqrt(3.d0))*y1*y2*s4a*s4b**2+ sqrt(3.d0)*(-      &
            2.d0*fea111344*sqrt(3.d0)-3.d0*hea233344-                           &
            3.d0*fea233344*sqrt(3.d0)-fea133344*sqrt(3.d0)+ 3.d0*hea133344+     &
            3.d0*fea233345)*y1*y3**3*s4a*s4b/9.d0-sqrt(3.d0)*(3.d0*fea133445-   &
            hea233444-3.d0*fea233445+ 3.d0*fea113444*sqrt(3.d0)-                &
            6.d0*sqrt(3.d0)*hea223555+ 5.d0*hea133444+                          &
            2.d0*fea122444*sqrt(3.d0))*y1**2*y3*s4a**2*s4b/3.d0                 
                 s6 = s4+ sqrt(3.d0)*(56.d0*sqrt(3.d0)*hea135555+               &
            69.d0*hea234555-9.d0*hea234445+ 216.d0*fea124444+                   &
            96.d0*fea235555)*y1*y3*s4a**2*s4b**2/72.d0-                         &
            sqrt(3.d0)*(3.d0*hea234555+ hea234445+                              &
            8.d0*fea124444)*y1*y3*s4a**4/8.d0+                                  &
            sqrt(3.d0)*(3.d0*fea24555*sqrt(3.d0)+ 3.d0*hea34555-                &
            4.d0*fea24455-3.d0*hea14445)*y2*s4a**2*s4b**2/4.d0+                 &
            sqrt(3.d0)*(fea23455*sqrt(3.d0)+ 2.d0*fea12555+                     &
            6.d0*hea13444)*y2*y3*s4b**3/3.d0                                    
                 s5 = s6-sqrt(3.d0)*(-4.d0*fea12455*sqrt(3.d0)+                 &
            fea23455*sqrt(3.d0)+ 18.d0*fea12555+                                &
            18.d0*hea13444)*y2*y3*s4a**2*s4b/9.d0-sqrt(3.d0)*(-                 &
            2.d0*fea133334*sqrt(3.d0)+ 3.d0*hea233334-                          &
            4.d0*fea233334*sqrt(3.d0)+ 3.d0*fea233335)*y1**4*y2*s4b/18.d0-      &
            sqrt(3.d0)*(3.d0*fea24555*sqrt(3.d0)+ 3.d0*hea34555-                &
            4.d0*fea24455-3.d0*hea14445)*y3*s4a**2*s4b**2/4.d0+                 &
            sqrt(3.d0)*(sqrt(3.d0)*hea223345+ 4.d0*hea113344-                   &
            2.d0*fea223344*sqrt(3.d0))*y1**2*y2**2*s4a*s4b/6.d0-sqrt(3.d0)*(-   &
            2.d0*fea133334*sqrt(3.d0)+ 3.d0*hea233334-                          &
            4.d0*fea233334*sqrt(3.d0)+ 3.d0*fea233335)*y1**4*y3*s4b/18.d0       
                 s1 = s5+ sqrt(3.d0)*(hea233344-fea133344*sqrt(3.d0)+           &
            hea133344-fea233344*sqrt(3.d0))*y1**3*y2*s4a*s4b/3.d0+              &
            sqrt(3.d0)*(-hea223334+ fea223334*sqrt(3.d0)+                       &
            hea111334)*y1**3*y2**2*s4b/3.d0+ hea223345*y2**2*y3**2*s4a*s4b+     &
            sqrt(3.d0)*(hea233344-fea133344*sqrt(3.d0)+ hea133344-              &
            fea233344*sqrt(3.d0))*y1**3*y3*s4a*s4b/3.d0+ sqrt(3.d0)*(-          &
            hea223334+ fea223334*sqrt(3.d0)+ hea111334)*y1**3*y3**2*s4b/3.d0-   &
            sqrt(3.d0)*fea11123*y1*y2*y3**3/2.d0-sqrt(3.d0)*(5.d0*fea12245+     &
            2.d0*fea11245+ 6.d0*hea11244+ 4.d0*fea22344*sqrt(3.d0)+             &
            2.d0*fea11255*sqrt(3.d0))*y1*y2**2*s4a*s4b/3.d0-                    &
            sqrt(3.d0)*(3.d0*hea13444-fea12455*sqrt(3.d0)+                      &
            fea12555)*y1*y2*s4b**3/3.d0+ hea123345*y1*y2**2*y3*s4a*s4b-         &
            sqrt(3.d0)*(5.d0*fea12245+ 2.d0*fea11245+ 6.d0*hea11244+            &
            4.d0*fea22344*sqrt(3.d0)+                                           &
            2.d0*fea11255*sqrt(3.d0))*y1*y3**2*s4a*s4b/3.d0                     
                 s5 = s1-sqrt(3.d0)*(2.d0*fea22235*sqrt(3.d0)-3.d0*fea11124-    &
            fea11135*sqrt(3.d0)+ 3.d0*hea13335)*y1*y3**3*s4a/9.d0-              &
            sqrt(3.d0)*(-6.d0*fea233445-6.d0*hea233444+                         &
            5.d0*fea122444*sqrt(3.d0)+ 4.d0*fea113444*sqrt(3.d0)-               &
            12.d0*sqrt(3.d0)*hea223555+ 9.d0*hea133444+                         &
            9.d0*fea133445)*y1*y2**2*s4b**3/9.d0-sqrt(3.d0)*(-6.d0*fea11124+    &
            fea11135*sqrt(3.d0)-2.d0*fea22235*sqrt(3.d0)+                       &
            6.d0*hea13335)*y1**3*y3*s4a/9.d0+ sqrt(3.d0)*(-6.d0*fea11124+       &
            fea11135*sqrt(3.d0)-2.d0*fea22235*sqrt(3.d0)+                       &
            6.d0*hea13335)*y1**3*y2*s4a/9.d0+                                   &
            2.d0/9.d0*sqrt(3.d0)*(3.d0*fea22245+ 2.d0*fea22255*sqrt(3.d0)+      &
            fea11155*sqrt(3.d0))*y1**3*s4a*s4b+ sqrt(3.d0)*(18.d0*hea33444+     &
            4.d0*fea22455*sqrt(3.d0)-fea11455*sqrt(3.d0)+                       &
            18.d0*fea22555)*y1**2*s4a**2*s4b/9.d0+                              &
            sqrt(3.d0)*fea11123*y1*y2**3*y3/2.d0+                               &
            sqrt(3.d0)*(2.d0*fea22235*sqrt(3.d0)-3.d0*fea11124-                 &
            fea11135*sqrt(3.d0)+ 3.d0*hea13335)*y1*y2**3*s4a/9.d0               
                 s4 = s5+ (2.d0/3.d0*fea233445+ hea233444+                      &
            2.d0/3.d0*fea122444*sqrt(3.d0)-2.d0/3.d0*fea113444*sqrt(3.d0)-      &
            2.d0/3.d0*fea133445+ 2.d0/3.d0*fea113445)*y2**2*y3*s4a*s4b**2+      &
            sqrt(3.d0)*fea123333*y1*y2*y3**4-sqrt(3.d0)*(2.d0*fea22555+         &
            6.d0*hea33444-fea11455*sqrt(3.d0))*y1**2*s4b**3/3.d0-               &
            sqrt(3.d0)*(3.d0*hea13444-fea12455*sqrt(3.d0)+                      &
            fea12555)*y1*y3*s4b**3/3.d0+ sqrt(3.d0)*(9.d0*hea13444+             &
            2.d0*fea23455*sqrt(3.d0)+ fea12455*sqrt(3.d0)+                      &
            9.d0*fea12555)*y1*y3*s4a**2*s4b/9.d0+                               &
            sqrt(3.d0)*(4.d0*fea22255*sqrt(3.d0)-3.d0*fea22245+                 &
            2.d0*fea11155*sqrt(3.d0))*y2**3*s4a*s4b/9.d0+                       &
            sqrt(3.d0)*(fea22235*sqrt(3.d0)-6.d0*fea11124+                      &
            4.d0*fea11135*sqrt(3.d0)+ 6.d0*hea13335)*y2**3*y3*s4a/9.d0-         &
            sqrt(3.d0)*(fea22235*sqrt(3.d0)-6.d0*fea11124+                      &
            4.d0*fea11135*sqrt(3.d0)+ 6.d0*hea13335)*y2*y3**3*s4a/9.d0-         &
            sqrt(3.d0)*(-fea22455*sqrt(3.d0)+ 9.d0*hea33444-                    &
            2.d0*fea11455*sqrt(3.d0)+ 9.d0*fea22555)*y2**2*s4a**2*s4b/9.d0      
                 s5 = s4+ (-2.d0*hea22335+ 3.d0*fea11224)*y1**2*y2**2*s4b-      &
            sqrt(3.d0)*(hea334555-3.d0*hea334445+ 8.d0*fea335555-               &
            2.d0*sqrt(3.d0)*hea334455-2.d0*hea114555)*y3**2*s4b**4/4.d0+        &
            hea122335*y1*y2**2*y3**2*s4b+ (-hea123344/3.d0-                     &
            sqrt(3.d0)*hea123345/3.d0-                                          &
            2.d0/3.d0*fea112344*sqrt(3.d0))*y1*y2*y3**2*s4b**2-                 &
            hea233344*y2**3*y3*s4a**2+ (-17.d0/8.d0*hea234555-                  &
            3.d0/8.d0*hea234445-9.d0*fea124444-                                 &
            5.d0/3.d0*sqrt(3.d0)*hea135555-2.d0*fea235555)*y1*y2*s4a*s4b**3+    &
            hea122333*y1*y2**2*y3**3+ (4.d0*hea12335-                           &
            3.d0*fea11234)*y1**2*y2*y3*s4b                                      
                 s6 = s5+ (5.d0/8.d0*hea234445-sqrt(3.d0)*hea135555/3.d0+       &
            7.d0/8.d0*hea234555+ 3.d0*fea124444+                                &
            2.d0*fea235555)*y1*y2*s4a**3*s4b-sqrt(3.d0)*(-fea122334+            &
            hea122335)*y1**2*y2**2*y3*s4a/4.d0+ (13.d0*hea355555-               &
            21.d0/5.d0*hea344445-18.d0/5.d0*hea144445-                          &
            9.d0/5.d0*sqrt(3.d0)*hea344455-14.d0/15.d0*sqrt(3.d0)*hea345555-    &
            16.d0/5.d0*fea145555)*y1*s4a**2*s4b**3+                             &
            sqrt(3.d0)*(6.d0*hea333355-sqrt(3.d0)*hea111145-                    &
            6.d0*fea333344*sqrt(3.d0))*y3**4*s4a*s4b/6.d0                       
                 s3 = s6+ sqrt(3.d0)*(-3.d0*sqrt(3.d0)*hea222555+               &
            5.d0*hea333444+ 4.d0*fea222444*sqrt(3.d0)+                          &
            3.d0*hea333455)*y3**3*s4a**2*s4b/3.d0+                              &
            sqrt(3.d0)*(8.d0*fea22344*sqrt(3.d0)+ 6.d0*fea11255*sqrt(3.d0)+     &
            4.d0*fea12255*sqrt(3.d0)+ 7.d0*fea11245+ 10.d0*fea12245+            &
            6.d0*hea11244)*y1**2*y3*s4a*s4b/3.d0+ hea223555*y2*y3**2*s4b**3+    &
            sqrt(3.d0)*(8.d0*fea22344*sqrt(3.d0)+ 6.d0*fea11255*sqrt(3.d0)+     &
            4.d0*fea12255*sqrt(3.d0)+ 7.d0*fea11245+ 10.d0*fea12245+            &
            6.d0*hea11244)*y1**2*y2*s4a*s4b/3.d0+ hea123335*y1*y2**3*y3*s4b+    &
            (fea11135*sqrt(3.d0)/3.d0+ fea11124+                                &
            fea22235*sqrt(3.d0)/3.d0)*y2**3*y3*s4b                              
                 s5 = s3+ sqrt(3.d0)*(-fea11114+ hea33335)*y2**4*s4a+           &
            (fea12255*sqrt(3.d0)+ fea11255*sqrt(3.d0)+ hea11244+                &
            2.d0*fea22344*sqrt(3.d0)+ 2.d0*fea11245+                            &
            2.d0*fea12245)*y1*y3**2*s4a**2+ (-fea133334*sqrt(3.d0)/3.d0+        &
            fea233334*sqrt(3.d0)/3.d0-fea233335)*y1*y2**4*s4a+ (-               &
            5.d0/3.d0*fea233445-hea233444+ 10.d0/3.d0*fea122444*sqrt(3.d0)+     &
            8.d0/3.d0*fea113444*sqrt(3.d0)+ 11.d0/3.d0*fea133445+               &
            fea113445/3.d0-6.d0*sqrt(3.d0)*hea223555+                           &
            3.d0*hea133444)*y1**2*y3*s4a*s4b**2+ hea114555*y1**2*s4a*s4b**3+    &
            (sqrt(3.d0)*hea223345/6.d0+ hea113344/3.d0+                         &
            fea223344*sqrt(3.d0)/3.d0)*y1**2*y3**2*s4b**2+ (2.d0*fea22344+      &
            2.d0*fea11255+ 2.d0*fea12255+ fea11245*sqrt(3.d0)/3.d0+             &
            fea12245*sqrt(3.d0)/3.d0)*y2**2*y3*s4a*s4b+ (5.d0/8.d0*hea234445-   &
            sqrt(3.d0)*hea135555/3.d0+ 7.d0/8.d0*hea234555+ 3.d0*fea124444+     &
            2.d0*fea235555)*y1*y3*s4a**3*s4b                                    
                 s6 = s5+ (fea12255*sqrt(3.d0)/3.d0+                            &
            5.d0/3.d0*fea11255*sqrt(3.d0)+ 2.d0*fea22344*sqrt(3.d0)+            &
            2.d0*fea11245+ 3.d0*fea12245+ 3.d0*hea11244)*y1*y2**2*s4b**2+       &
            (5.d0/2.d0*hea14445+ 4.d0*fea15555-7.d0/2.d0*hea34555-              &
            11.d0/6.d0*fea24555*sqrt(3.d0)+ 4.d0*fea24455)*y2*s4a**3*s4b+       &
            (hea123344/3.d0+ sqrt(3.d0)*hea123345/3.d0+                         &
            2.d0/3.d0*fea112344*sqrt(3.d0))*y1*y2**2*y3*s4b**2+                 &
            (7.d0*hea355555-9.d0/5.d0*hea344445-12.d0/5.d0*hea144445-           &
            6.d0/5.d0*sqrt(3.d0)*hea344455+ 4.d0/15.d0*sqrt(3.d0)*hea345555-    &
            4.d0/5.d0*fea145555)*y3*s4a**2*s4b**3                               
                 s4 = s6+ (-2.d0/9.d0*fea111344*sqrt(3.d0)+ hea233344/3.d0+     &
            fea233344*sqrt(3.d0)/3.d0-fea133344*sqrt(3.d0)/9.d0+                &
            fea233345/3.d0)*y1*y3**3*s4b**2+ (-2.d0/3.d0*fea113445-             &
            10.d0/3.d0*fea133445+ 4.d0/3.d0*fea233445-3.d0*hea133444+           &
            2.d0*hea233444+ 6.d0*sqrt(3.d0)*hea223555-                          &
            10.d0/3.d0*fea113444*sqrt(3.d0)-                                    &
            8.d0/3.d0*fea122444*sqrt(3.d0))*y1*y2**2*s4a*s4b**2+ (-             &
            sqrt(3.d0)*fea122333/2.d0-hea122333/2.d0)*y1**2*y2**3*y3+           &
            (5.d0/2.d0*hea14445+ 4.d0*fea15555-7.d0/2.d0*hea34555-              &
            11.d0/6.d0*fea24555*sqrt(3.d0)+ 4.d0*fea24455)*y3*s4a**3*s4b+       &
            (5.d0/3.d0*fea233445+ hea233444-10.d0/3.d0*fea122444*sqrt(3.d0)-    &
            8.d0/3.d0*fea113444*sqrt(3.d0)-11.d0/3.d0*fea133445-                &
            fea113445/3.d0+ 6.d0*sqrt(3.d0)*hea223555-                          &
            3.d0*hea133444)*y1**2*y2*s4a*s4b**2                                 
                 s5 = s4-sqrt(3.d0)*fea123333*y1*y2**4*y3+                      &
            hea234445*y2*y3*s4a**3*s4b+ (hea233334/2.d0+                        &
            fea133334*sqrt(3.d0)/3.d0-fea233334*sqrt(3.d0)/3.d0-                &
            fea233335/2.d0)*y1**4*y3*s4a+ (2.d0/9.d0*fea133344*sqrt(3.d0)+      &
            fea233345/3.d0-fea233344*sqrt(3.d0)/3.d0+ hea233344/3.d0+           &
            fea111344*sqrt(3.d0)/9.d0)*y1**3*y2*s4b**2+                         &
            (fea11135*sqrt(3.d0)/3.d0+ fea11124+                                &
            fea22235*sqrt(3.d0)/3.d0)*y2*y3**3*s4b+ (-3.d0*hea33444-            &
            2.d0/3.d0*fea22455*sqrt(3.d0)+                                      &
            2.d0/3.d0*fea11455*sqrt(3.d0))*y2**2*s4a*s4b**2+                    &
            (2.d0/3.d0*fea113445+ 10.d0/3.d0*fea133445-4.d0/3.d0*fea233445+     &
            3.d0*hea133444-2.d0*hea233444-6.d0*sqrt(3.d0)*hea223555+            &
            10.d0/3.d0*fea113444*sqrt(3.d0)+                                    &
            8.d0/3.d0*fea122444*sqrt(3.d0))*y1*y3**2*s4a*s4b**2+                &
            sqrt(3.d0)*(3.d0*hea234555+ hea234445+                              &
            8.d0*fea124444)*y1*y2*s4a**4/8.d0+ (hea11244+                       &
            fea22344*sqrt(3.d0)+ fea11255*sqrt(3.d0)+ fea11245+                 &
            fea12245)*y2**2*y3*s4a**2                                           
                 s2 = s5+ hea234555*y2*y3*s4a*s4b**3+                           &
            sqrt(3.d0)*(12.d0*hea34555-15.d0*fea24455+                          &
            5.d0*fea24555*sqrt(3.d0)-6.d0*fea15555-                             &
            9.d0*hea14445)*y2*s4a**4/18.d0+ hea144445*y1*s4a**4*s4b+            &
            (3.d0*hea33444+ 2.d0/3.d0*fea22455*sqrt(3.d0)-                      &
            2.d0/3.d0*fea11455*sqrt(3.d0))*y3**2*s4a*s4b**2+                    &
            (fea24555*sqrt(3.d0)+ hea34555)*y1*s4a*s4b**3+ (hea122333/2.d0-     &
            sqrt(3.d0)*fea122333/2.d0)*y1**3*y2**2*y3-                          &
            sqrt(3.d0)*(5.d0*hea34555-6.d0*fea24455+                            &
            3.d0*fea24555*sqrt(3.d0)-4.d0*fea15555-                             &
            3.d0*hea14445)*y3*s4b**4/4.d0+ hea55555*s4b**5-                     &
            sqrt(3.d0)*(fea22255-fea11155)*y2**3*s4a**2/3.d0+                   &
            (2.d0/3.d0*fea11135*sqrt(3.d0)-fea11124+                            &
            2.d0/3.d0*fea22235*sqrt(3.d0)+ 2.d0*hea13335)*y1**3*y3*s4b          
                 s5 = s2+ (-17.d0/8.d0*hea234555-3.d0/8.d0*hea234445-           &
            9.d0*fea124444-5.d0/3.d0*sqrt(3.d0)*hea135555-                      &
            2.d0*fea235555)*y1*y3*s4a*s4b**3+ sqrt(3.d0)*(-9.d0*hea334445-      &
            14.d0*hea114555+ 11.d0*hea334555+ 24.d0*fea335555-                  &
            10.d0*sqrt(3.d0)*hea334455)*y2**2*s4a**4/36.d0-sqrt(3.d0)*(-        &
            3.d0*hea333444-6.d0*fea222444*sqrt(3.d0)+                           &
            4.d0*sqrt(3.d0)*hea222555-3.d0*hea333455)*y1**3*s4b**3/6.d0+        &
            hea113334*y1**2*y3**3*s4a+ (-fea233445-hea233444+ hea133444+        &
            fea133445-fea113445)*y1**2*y3*s4a**3+ sqrt(3.d0)*(hea334555-        &
            3.d0*hea334445+ 8.d0*fea335555-2.d0*sqrt(3.d0)*hea334455-           &
            2.d0*hea114555)*y2**2*s4b**4/4.d0-sqrt(3.d0)*(-9.d0*hea334445-      &
            14.d0*hea114555+ 11.d0*hea334555+ 24.d0*fea335555-                  &
            10.d0*sqrt(3.d0)*hea334455)*y3**2*s4a**4/36.d0+ (-                  &
            fea12255*sqrt(3.d0)/3.d0-5.d0/3.d0*fea11255*sqrt(3.d0)-             &
            2.d0*fea22344*sqrt(3.d0)-2.d0*fea11245-3.d0*fea12245-               &
            3.d0*hea11244)*y1*y3**2*s4b**2                                      
                 s4 = s5+ (-fea12255*sqrt(3.d0)-fea11255*sqrt(3.d0)-hea11244-   &
            2.d0*fea22344*sqrt(3.d0)-2.d0*fea11245-                             &
            2.d0*fea12245)*y1*y2**2*s4a**2+ (-3.d0*hea11244-                    &
            4.d0*fea22344*sqrt(3.d0)-8.d0/3.d0*fea11255*sqrt(3.d0)-             &
            4.d0/3.d0*fea12255*sqrt(3.d0)-3.d0*fea11245-                        &
            4.d0*fea12245)*y1**2*y2*s4b**2+ (3.d0*hea11244+                     &
            4.d0*fea22344*sqrt(3.d0)+ 8.d0/3.d0*fea11255*sqrt(3.d0)+            &
            4.d0/3.d0*fea12255*sqrt(3.d0)+ 3.d0*fea11245+                       &
            4.d0*fea12245)*y1**2*y3*s4b**2+ (-hea123335/2.d0+                   &
            3.d0/2.d0*fea123334)*y1**3*y2*y3*s4b+                               &
            (2.d0/3.d0*fea12255*sqrt(3.d0)+ 2.d0*fea11245+ 3.d0*hea11244+       &
            3.d0*fea22344*sqrt(3.d0)+ 7.d0/3.d0*fea11255*sqrt(3.d0)+            &
            4.d0*fea12245)*y2**2*y3*s4b**2+ sqrt(3.d0)*(fea22255-               &
            fea11155)*y3**3*s4a**2/3.d0+ hea135555*y1*y3*s4b**4+                &
            sqrt(3.d0)*(3.d0*hea33444+ fea22455*sqrt(3.d0)+                     &
            fea22555)*y2**2*s4b**3/3.d0-sqrt(3.d0)*(hea333335-                  &
            fea333334)*y2**5*s4a/2.d0                                           
                 s3 = s4-sqrt(3.d0)*(12.d0*hea34555-15.d0*fea24455+             &
            5.d0*fea24555*sqrt(3.d0)-6.d0*fea15555-                             &
            9.d0*hea14445)*y3*s4a**4/18.d0+ hea133344*y1*y3**3*s4a**2+          &
            hea13444*y1*y3*s4a**3-hea13444*y1*y2*s4a**3+                        &
            hea13335*y1*y3**3*s4b+ hea111334*y1**3*y3**2*s4a+                   &
            hea133444*y1*y3**2*s4a**3+ hea113344*y1**2*y3**2*s4a**2+            &
            hea344455*y3*s4a**3*s4b**2+ hea233344*y2*y3**3*s4a**2+              &
            (2.d0*hea223333+ sqrt(3.d0)*fea111122)*y1**4*y2**2+                 &
            hea111145*y1**4*s4a*s4b+ hea11244*y1**2*y2*s4a**2+                  &
            hea345555*y3*s4a*s4b**4+ hea344445*y2*s4a**4*s4b+                   &
            hea34555*y3*s4a*s4b**3+ (3.d0/4.d0*fea122334+                       &
            hea122335/4.d0)*y1**2*y2**2*y3*s4b+ hea333455*y3**3*s4a*s4b**2      
                 s5 = s3+ (fea133334*sqrt(3.d0)/3.d0-                           &
            fea233334*sqrt(3.d0)/3.d0+ fea233335)*y1*y3**4*s4a-                 &
            sqrt(3.d0)*(hea111145+ 2.d0*fea333344)*y2**4*s4a**2/2.d0+           &
            hea344445*y3*s4a**4*s4b+ hea34555*y2*s4a*s4b**3+                    &
            hea12335*y1*y2**2*y3*s4b+ (4.d0/9.d0*fea133344*sqrt(3.d0)-          &
            4.d0/9.d0*fea111344*sqrt(3.d0)-fea233345/3.d0-                      &
            hea233344/3.d0)*y2*y3**3*s4b**2+ hea13335*y1*y2**3*s4b+             &
            (2.d0/3.d0*fea11135*sqrt(3.d0)-fea11124+                            &
            2.d0/3.d0*fea22235*sqrt(3.d0)+ 2.d0*hea13335)*y1**3*y2*s4b          
                 s4 = s5+ hea14445*y1*s4a**3*s4b+ (2.d0*fea22344+               &
            2.d0*fea11255+ 2.d0*fea12255+ fea11245*sqrt(3.d0)/3.d0+             &
            fea12245*sqrt(3.d0)/3.d0)*y2*y3**2*s4a*s4b+                         &
            (fea133344*sqrt(3.d0)/3.d0-fea111344*sqrt(3.d0)/3.d0-hea133344-     &
            fea233345)*y1**3*y2*s4a**2+ (-hea11244-fea22344*sqrt(3.d0)-         &
            fea11255*sqrt(3.d0)-fea11245-fea12245)*y2*y3**2*s4a**2-             &
            sqrt(3.d0)*(-fea11114+ hea33335)*y3**4*s4a+                         &
            (sqrt(3.d0)*fea122333/2.d0+ hea122333/2.d0)*y1**2*y2*y3**3+ (-      &
            sqrt(3.d0)*hea223345/6.d0-hea113344/3.d0-                           &
            fea223344*sqrt(3.d0)/3.d0)*y1**2*y2**2*s4b**2+                      &
            sqrt(3.d0)*(3.d0*hea233334+ fea133334*sqrt(3.d0)+                   &
            2.d0*fea233334*sqrt(3.d0)+ 3.d0*fea233335)*y1*y3**4*s4b/9.d0+       &
            hea334555*y3**2*s4a*s4b**3-hea11244*y1**2*y3*s4a**2                 
                 s5 = s4+ sqrt(3.d0)*(hea333335-fea333334)*y3**5*s4a/2.d0-      &
            sqrt(3.d0)*(28.d0*hea144445+ 26.d0*hea344445+                       &
            19.d0*sqrt(3.d0)*hea344455+ 3.d0*sqrt(3.d0)*hea345555-              &
            90.d0*hea355555+ 36.d0*fea145555)*y2*s4a**5/75.d0+                  &
            hea12335*y1*y2*y3**2*s4b-sqrt(3.d0)*(hea12335-                      &
            fea11234)*y1*y2*y3**2*s4a+ sqrt(3.d0)*(hea22335-                    &
            fea11224)*y1**2*y3**2*s4a+ sqrt(3.d0)*(hea12335-                    &
            fea11234)*y1*y2**2*y3*s4a-sqrt(3.d0)*(2.d0*fea133344*sqrt(3.d0)+    &
            4.d0*fea111344*sqrt(3.d0)+ 6.d0*hea133344+                          &
            3.d0*fea233345)*y2*y3**3*s4a*s4b/9.d0+ sqrt(3.d0)*(hea111145+       &
            2.d0*fea333344)*y3**4*s4a**2/2.d0+ (-2.d0*hea22335+                 &
            3.d0*fea11224)*y1**2*y3**2*s4b                                      
                 s6 = s5+ (-hea122333/2.d0+                                     &
            sqrt(3.d0)*fea122333/2.d0)*y1**3*y2*y3**2+                          &
            sqrt(3.d0)*(28.d0*hea144445+ 26.d0*hea344445+                       &
            19.d0*sqrt(3.d0)*hea344455+ 3.d0*sqrt(3.d0)*hea345555-              &
            90.d0*hea355555+ 36.d0*fea145555)*y3*s4a**5/75.d0+                  &
            (3.d0*hea13444+ 2.d0/3.d0*fea23455*sqrt(3.d0)-                      &
            2.d0/3.d0*fea12455*sqrt(3.d0))*y1*y3*s4a*s4b**2+ (-                 &
            2.d0/9.d0*fea133344*sqrt(3.d0)-fea233345/3.d0+                      &
            fea233344*sqrt(3.d0)/3.d0-hea233344/3.d0-                           &
            fea111344*sqrt(3.d0)/9.d0)*y1**3*y3*s4b**2                           
                 t56y= s6+ (-fea133344*sqrt(3.d0)/3.d0+                         &
            fea111344*sqrt(3.d0)/3.d0+ hea133344+ fea233345)*y1**3*y3*s4a**2-   &
            sqrt(3.d0)*(-3.d0*fea233445-3.d0*hea233444+                         &
            4.d0*fea122444*sqrt(3.d0)+ 5.d0*fea113444*sqrt(3.d0)-               &
            12.d0*sqrt(3.d0)*hea223555+ 9.d0*hea133444+                         &
            9.d0*fea133445)*y1**2*y3*s4b**3/9.d0+ sqrt(3.d0)*(hea113334+        &
            hea223334+ fea223334*sqrt(3.d0))*y1**2*y3**3*s4b/3.d0-              &
            sqrt(3.d0)*(hea22335-fea11224)*y1**2*y2**2*s4a+                     &
            sqrt(3.d0)*fea11223*y1**2*y2**2*y3+ sqrt(3.d0)*(hea123335-          &
            fea123334)*y1*y2*y3**3*s4a/2.d0                                      
          !
         endif
          !                                                           
          dipol_xy = ( t4y+ t56y )                                       
          !
        end select 
        !
        dipol_xy = dipol_xy*factor
        !
    end function dipol_xy
   
   
   double precision function DMS_A(parmax,param,local)
   
    integer,intent(in)          ::  parmax
    double precision,intent(in) ::  param(parmax)
    double precision,intent(in) ::  local(7)
   
    double precision            ::  r14,r24,r34,alpha1,alpha2,alpha3
    double precision            ::  y1,y2,y3,y4,y5,alpha,rho
   
    double precision            ::  v,v0,rhoe,pi,v1,v2,v3,v4,v5,v6
   
    double precision            ::  sinrho,drho,cosrho,beta,de,b0
   
                           
    double precision                                        &
         fea    ,fea1  ,                                    &
         fea11  ,fea12  ,fea14  ,fea44  ,                   &
         fea111 ,fea112 ,fea114 ,fea123 ,                   &
         fea124 ,fea144 ,fea155 ,fea455 ,                   &
         fea1111,fea1112,fea1114,fea1122,                   &
         fea1123,fea1124,fea1125,fea1144,                   &
         fea1155,fea1244,fea1255,fea1444,                   &
         fea1455,fea4444
         double precision ::   &
         fea44444 ,fea33455 ,fea33445 ,fea33345 ,fea33344 ,&
         fea33334 ,fea33333 ,fea25555 ,fea24455 ,fea24445 ,fea23333 ,&
         fea13455 ,fea13445 ,fea13345 ,fea12355 ,fea11334 ,fea11333 ,&
         fea11255 ,fea11245 ,fea11234 ,fea11233 ,fea11135 ,fea11134 ,&
         fea11123 ,fea555555,fea444444,fea335555,fea334455,fea334445,&
         fea333555,fea333333,fea244555,fea244455,fea233445,fea233444,&
         fea233345,fea233344,fea233335,fea223355,fea222335,fea222334,&
         fea222333,fea222255,fea222245,fea222233,fea222224,fea145555,&
         fea134444,fea133444,fea133345,fea133334,fea133333,fea124555,&
         fea124455,fea123455,fea123345,fea113555,fea113345,fea112355,&
         fea112335,fea112233,fea111444,fea111234,fea111233,fea111123
   
    double precision :: s1,s2,s3,s4,s5,s6,rhobar
                       
    double precision :: &      
         Rhoedg,re14,aa1,ve  ,f0a,  &
         f1a,f2a,f3a,f4a,f5a,f6a,f7a,f8a, &
         f1a1,f2a1,f3a1,f4a1,f5a1,f6a1,  &
         f0a11,f1a11,f2a11,f3a11,f4a11, &
         f0a12,f1a12,f2a12,f3a12,f4a12, &
         f0a14,f1a14,f2a14,f3a14,f4a14, &
         f0a44,f1a44,f2a44,f3a44,f4a44, &
         f0a111,f1a111,f2a111,f3a111  , &
         f0a112,f1a112,f2a112,f3a112  , &
         f0a114,f1a114,f2a114,f3a114  , &
         f0a123,f1a123,f2a123,f3a123  , &
         f0a124,f1a124,f2a124,f3a124  , &
         f0a144,f1a144,f2a144,f3a144  , &
         f0a155,f1a155,f2a155,f3a155  , &
         f0a455,f1a455,f2a455,f3a455  , &
         f0a1111,f1a1111,f2a1111      , &
         f0a1112,f1a1112,f2a1112      , &
         f0a1114,f1a1114,f2a1114      , &
         f0a1122,f1a1122,f2a1122      , &
         f0a1123,f1a1123,f2a1123      , &
         f0a1124,f1a1124,f2a1124      , &
         f0a1125,f1a1125,f2a1125      , &
         f0a1144,f1a1144,f2a1144      , &
         f0a1155,f1a1155,f2a1155      , &
         f0a1244,f1a1244,f2a1244      , &
         f0a1255,f1a1255,f2a1255      , &
         f0a1444,f1a1444,f2a1444      , &
         f0a1455,f1a1455,f2a1455      , &
         f0a4444,f1a4444,f2a4444      , &
         f0a44444 ,f1a44444 ,           &
         f2a44444 ,f0a33455 ,f1a33455 ,f2a33455 ,f0a33445 ,f1a33445 ,&
         f2a33445 ,f0a33345 ,f1a33345 ,f2a33345 ,f0a33344 ,f1a33344 ,&
         f2a33344 ,f0a33334 ,f1a33334 ,f2a33334 ,f0a33333 ,f1a33333 ,&
         f2a33333 ,f0a25555 ,f1a25555 ,f2a25555 ,f0a24455 ,f1a24455 ,&
         f2a24455 ,f0a24445 ,f1a24445 ,f2a24445 ,f0a23333 ,f1a23333 ,&
         f2a23333 ,f0a13455 ,f1a13455 ,f2a13455 ,f0a13445 ,f1a13445 ,&
         f2a13445 ,f0a13345 ,f1a13345 ,f2a13345 ,f0a12355 ,f1a12355 ,&
         f2a12355 ,f0a11334 ,f1a11334 ,f2a11334 ,f0a11333 ,f1a11333 ,&
         f2a11333 ,f0a11255 ,f1a11255 ,f2a11255 ,f0a11245 ,f1a11245 ,&
         f2a11245 ,f0a11234 ,f1a11234 ,f2a11234 ,f0a11233 ,f1a11233 ,&
         f2a11233 ,f0a11135 ,f1a11135 ,f2a11135 ,f0a11134 ,f1a11134 ,&
         f2a11134 ,f0a11123 ,f1a11123 ,f2a11123 ,f0a555555,f1a555555,&
         f2a555555,f0a444444,f1a444444,f2a444444,f0a335555,f1a335555,&
         f2a335555,f0a334455,f1a334455,f2a334455,f0a334445,f1a334445,&
         f2a334445,f0a333555,f1a333555,f2a333555,f0a333333,f1a333333,&
         f2a333333,f0a244555,f1a244555,f2a244555,f0a244455,f1a244455,&
         f2a244455,f0a233445,f1a233445,f2a233445,f0a233444,f1a233444,&
         f2a233444,f0a233345,f1a233345,f2a233345,f0a233344,f1a233344,&
         f2a233344,f0a233335,f1a233335,f2a233335,f0a223355,f1a223355,&
         f2a223355,f0a222335,f1a222335,f2a222335,f0a222334,f1a222334,&
         f2a222334,f0a222333,f1a222333,f2a222333,f0a222255,f1a222255,&
         f2a222255,f0a222245,f1a222245,f2a222245,f0a222233,f1a222233,&
         f2a222233,f0a222224,f1a222224,f2a222224,f0a145555,f1a145555,&
         f2a145555,f0a134444,f1a134444,f2a134444,f0a133444,f1a133444,&
         f2a133444,f0a133345,f1a133345,f2a133345,f0a133334,f1a133334,&
         f2a133334,f0a133333,f1a133333,f2a133333,f0a124555,f1a124555,&
         f2a124555,f0a124455,f1a124455,f2a124455,f0a123455,f1a123455,&
         f2a123455,f0a123345,f1a123345,f2a123345,f0a113555,f1a113555,&
         f2a113555,f0a113345,f1a113345,f2a113345,f0a112355,f1a112355,&
         f2a112355,f0a112335,f1a112335,f2a112335,f0a112233,f1a112233,&
         f2a112233,f0a111444,f1a111444,f2a111444,f0a111234,f1a111234,&
         f2a111234,f0a111233,f1a111233,f2a111233,f0a111123,f1a111123,&
         f2a111123
   
   
   
     !-------------------------------
   
   
      pi=3.141592653589793
   
      rhoedg     = param(  1)
      rhoe=pi*rhoedg/1.8d+02 
   
      re14       = param( 2)
      b0         = param( 3)**2
      de         = param( 4)
   
      f1a        = param(  5)
      f2a        = param(  6)
      f3a        = param(  7)
      f4a        = param(  8)
      f5a        = param(  9)
      f6a        = param( 10)
      f7a        = param( 11)
      !
      f0a        = param( 12)
      f1a1       = param( 13)
      f2a1       = param( 14)
      f3a1       = param( 15)
      f4a1       = param( 16)
      f5a1       = param( 17)
      f6a1       = param( 18)
      f0a11      = param( 19)
      f1a11      = param( 20)
      f2a11      = param( 21)
      f3a11      = param( 22)
      f4a11      = param( 23)
      f0a12      = param( 24)
      f1a12      = param( 25)
      f2a12      = param( 26)
      f3a12      = param( 27)
      f4a12      = param( 28)
      f0a14      = param( 29)
      f1a14      = param( 30)
      f2a14      = param( 31)
      f3a14      = param( 32)
      f4a14      = param( 33)
      f0a44      = param( 34)
      f1a44      = param( 35)
      f2a44      = param( 36)
      f3a44      = param( 37)
      f4a44      = param( 38)
      f0a111     = param( 39)
      f1a111     = param( 40)
      f2a111     = param( 41)
      f3a111     = param( 42)
      f0a112     = param( 43)
      f1a112     = param( 44)
      f2a112     = param( 45)
      f3a112     = param( 46)
      f0a114     = param( 47)
      f1a114     = param( 48)
      f2a114     = param( 49)
      f3a114     = param( 50)
      f0a123     = param( 51)
      f1a123     = param( 52)
      f2a123     = param( 53)
      f3a123     = param( 54)
      f0a124     = param( 55)
      f1a124     = param( 56)
      f2a124     = param( 57)
      f3a124     = param( 58)
      f0a144     = param( 59)
      f1a144     = param( 60)
      f2a144     = param( 61)
      f3a144     = param( 62)
      f0a155     = param( 63)
      f1a155     = param( 64)
      f2a155     = param( 65)
      f3a155     = param( 66)
      f0a455     = param( 67)
      f1a455     = param( 68)
      f2a455     = param( 69)
      f3a455     = param( 70)
      f0a1111    = param( 71)
      f1a1111    = param( 72)
      f2a1111    = param( 73)
      f0a1112    = param( 74)
      f1a1112    = param( 75)
      f2a1112    = param( 76)
      f0a1114    = param( 77)
      f1a1114    = param( 78)
      f2a1114    = param( 79)
      f0a1122    = param( 80)
      f1a1122    = param( 81)
      f2a1122    = param( 82)
      f0a1123    = param( 83)
      f1a1123    = param( 84)
      f2a1123    = param( 85)
      f0a1124    = param( 86)
      f1a1124    = param( 87)
      f2a1124    = param( 88)
      f0a1125    = param( 89)
      f1a1125    = param( 90)
      f2a1125    = param( 91)
      f0a1144    = param( 92)
      f1a1144    = param( 93)
      f2a1144    = param( 94)
      f0a1155    = param( 95)
      f1a1155    = param( 96)
      f2a1155    = param( 97)
      f0a1244    = param( 98)
      f1a1244    = param( 99)
      f2a1244    = param(100)
      f0a1255    = param(101)
      f1a1255    = param(102)
      f2a1255    = param(103)
      f0a1444    = param(104)
      f1a1444    = param(105)
      f2a1444    = param(106)
      f0a1455    = param(107)
      f1a1455    = param(108)
      f2a1455    = param(109)
      f0a4444    = param(110)
      f1a4444    = param(111)
      f2a4444    = param(112)
      !
      if (parmax>112) then 
        !
        f0a44444   = param(113)
        f1a44444   = param(114)
        f2a44444   = param(115)
        f0a33455   = param(116)
        f1a33455   = param(117)
        f2a33455   = param(118)
        f0a33445   = param(119)
        f1a33445   = param(120)
        f2a33445   = param(121)
        f0a33345   = param(122)
        f1a33345   = param(123)
        f2a33345   = param(124)
        f0a33344   = param(125)
        f1a33344   = param(126)
        f2a33344   = param(127)
        f0a33334   = param(128)
        f1a33334   = param(129)
        f2a33334   = param(130)
        f0a33333   = param(131)
        f1a33333   = param(132)
        f2a33333   = param(133)
        f0a25555   = param(134)
        f1a25555   = param(135)
        f2a25555   = param(136)
        f0a24455   = param(137)
        f1a24455   = param(138)
        f2a24455   = param(139)
        f0a24445   = param(140)
        f1a24445   = param(141)
        f2a24445   = param(142)
        f0a23333   = param(143)
        f1a23333   = param(144)
        f2a23333   = param(145)
        f0a13455   = param(146)
        f1a13455   = param(147)
        f2a13455   = param(148)
        f0a13445   = param(149)
        f1a13445   = param(150)
        f2a13445   = param(151)
        f0a13345   = param(152)
        f1a13345   = param(153)
        f2a13345   = param(154)
        f0a12355   = param(155)
        f1a12355   = param(156)
        f2a12355   = param(157)
        f0a11334   = param(158)
        f1a11334   = param(159)
        f2a11334   = param(160)
        f0a11333   = param(161)
        f1a11333   = param(162)
        f2a11333   = param(163)
        f0a11255   = param(164)
        f1a11255   = param(165)
        f2a11255   = param(166)
        f0a11245   = param(167)
        f1a11245   = param(168)
        f2a11245   = param(169)
        f0a11234   = param(170)
        f1a11234   = param(171)
        f2a11234   = param(172)
        f0a11233   = param(173)
        f1a11233   = param(174)
        f2a11233   = param(175)
        f0a11135   = param(176)
        f1a11135   = param(177)
        f2a11135   = param(178)
        f0a11134   = param(179)
        f1a11134   = param(180)
        f2a11134   = param(181)
        f0a11123   = param(182)
        f1a11123   = param(183)
        f2a11123   = param(184)
        f0a555555  = param(185)
        f1a555555  = param(186)
        f2a555555  = param(187)
        f0a444444  = param(188)
        f1a444444  = param(189)
        f2a444444  = param(190)
        f0a335555  = param(191)
        f1a335555  = param(192)
        f2a335555  = param(193)
        f0a334455  = param(194)
        f1a334455  = param(195)
        f2a334455  = param(196)
        f0a334445  = param(197)
        f1a334445  = param(198)
        f2a334445  = param(199)
        f0a333555  = param(200)
        f1a333555  = param(201)
        f2a333555  = param(202)
        f0a333333  = param(203)
        f1a333333  = param(204)
        f2a333333  = param(205)
        f0a244555  = param(206)
        f1a244555  = param(207)
        f2a244555  = param(208)
        f0a244455  = param(209)
        f1a244455  = param(210)
        f2a244455  = param(211)
        f0a233445  = param(212)
        f1a233445  = param(213)
        f2a233445  = param(214)
        f0a233444  = param(215)
        f1a233444  = param(216)
        f2a233444  = param(217)
        f0a233345  = param(218)
        f1a233345  = param(219)
        f2a233345  = param(220)
        f0a233344  = param(221)
        f1a233344  = param(222)
        f2a233344  = param(223)
        f0a233335  = param(224)
        f1a233335  = param(225)
        f2a233335  = param(226)
        f0a223355  = param(227)
        f1a223355  = param(228)
        f2a223355  = param(229)
        f0a222335  = param(230)
        f1a222335  = param(231)
        f2a222335  = param(232)
        f0a222334  = param(233)
        f1a222334  = param(234)
        f2a222334  = param(235)
        f0a222333  = param(236)
        f1a222333  = param(237)
        f2a222333  = param(238)
        f0a222255  = param(239)
        f1a222255  = param(240)
        f2a222255  = param(241)
        f0a222245  = param(242)
        f1a222245  = param(243)
        f2a222245  = param(244)
        f0a222233  = param(245)
        f1a222233  = param(246)
        f2a222233  = param(247)
        f0a222224  = param(248)
        f1a222224  = param(249)
        f2a222224  = param(250)
        f0a145555  = param(251)
        f1a145555  = param(252)
        f2a145555  = param(253)
        f0a134444  = param(254)
        f1a134444  = param(255)
        f2a134444  = param(256)
        f0a133444  = param(257)
        f1a133444  = param(258)
        f2a133444  = param(259)
        f0a133345  = param(260)
        f1a133345  = param(261)
        f2a133345  = param(262)
        f0a133334  = param(263)
        f1a133334  = param(264)
        f2a133334  = param(265)
        f0a133333  = param(266)
        f1a133333  = param(267)
        f2a133333  = param(268)
        f0a124555  = param(269)
        f1a124555  = param(270)
        f2a124555  = param(271)
        f0a124455  = param(272)
        f1a124455  = param(273)
        f2a124455  = param(274)
        f0a123455  = param(275)
        f1a123455  = param(276)
        f2a123455  = param(277)
        f0a123345  = param(278)
        f1a123345  = param(279)
        f2a123345  = param(280)
        f0a113555  = param(281)
        f1a113555  = param(282)
        f2a113555  = param(283)
        f0a113345  = param(284)
        f1a113345  = param(285)
        f2a113345  = param(286)
        f0a112355  = param(287)
        f1a112355  = param(288)
        f2a112355  = param(289)
        f0a112335  = param(290)
        f1a112335  = param(291)
        f2a112335  = param(292)
        f0a112233  = param(293)
        f1a112233  = param(294)
        f2a112233  = param(295)
        f0a111444  = param(296)
        f1a111444  = param(297)
        f2a111444  = param(298)
        f0a111234  = param(299)
        f1a111234  = param(300)
        f2a111234  = param(301)
        f0a111233  = param(302)
        f1a111233  = param(303)
        f2a111233  = param(304)
        f0a111123  = param(305)
        f1a111123  = param(306)
        f2a111123  = param(307)
        !
      endif 
   
      r14    = local(1) ;  r24    = local(2) ;  r34    = local(3)
      alpha1 = local(4) ;  alpha2 = local(5) ;  alpha3 = local(6)
      rhobar = local(7)
   
      pi=3.141592653589793
      rhoe=pi*rhoedg/1.8d+02
   
      y4=(2.d0*alpha1-alpha2-alpha3)/sqrt(6.d0)
      y5=(alpha2-alpha3)/sqrt(2.d0)
   
      alpha=(alpha1+alpha2+alpha3)/3.d0
      rho=pi-dasin(2.d0*sin(alpha*0.5d0)/dsqrt(3.d0))
   
      if ( 2.d0*sin(alpha*0.5d0)/dsqrt(3.d0).ge.1.0d0 ) then 
        sinrho=1.d0 
      else 
        sinrho = 2.d0*sin(alpha*0.5d0)/dsqrt(3.d0)
      endif 
      !
      cosrho = sqrt(1.d0-sinrho**2)*sign(1.0d0,cos(local(7)))
      !
      drho=(sin(rhoe)-sinrho)
      !
      y1=1.0d0*(r14-re14)*exp(-b0*(r14-re14)**2)
      y2=1.0d0*(r24-re14)*exp(-b0*(r24-re14)**2)
      y3=1.0d0*(r34-re14)*exp(-b0*(r34-re14)**2)
      !
      v0=de+f1a*drho+f2a*drho**2+f3a*drho**3+f4a*drho**4+f5a*drho**5 &
            +f6a*drho**6+f7a*drho**7
   
      fea1= f0a + f1a1*drho+f2a1*drho**2+f3a1*drho**3+f4a1*drho**4+f5a1*drho**5+f6a1*drho**6
   
      fea11=   f0a11+f1a11*drho+f2a11*drho**2+f3a11*drho**3+f4a11*drho**4
      fea12=   f0a12+f1a12*drho+f2a12*drho**2+f3a12*drho**3+f4a12*drho**4
      fea14=   f0a14+f1a14*drho+f2a14*drho**2+f3a14*drho**3+f4a14*drho**4
      fea44=   f0a44+f1a44*drho+f2a44*drho**2+f3a44*drho**3+f4a44*drho**4
    
      fea111= f0a111+f1a111*drho+f2a111*drho**2+f3a111*drho**3
      fea112= f0a112+f1a112*drho+f2a112*drho**2+f3a112*drho**3
      fea114= f0a114+f1a114*drho+f2a114*drho**2+f3a114*drho**3
      fea123= f0a123+f1a123*drho+f2a123*drho**2+f3a123*drho**3
      fea124= f0a124+f1a124*drho+f2a124*drho**2+f3a124*drho**3
      fea144= f0a144+f1a144*drho+f2a144*drho**2+f3a144*drho**3
      fea155= f0a155+f1a155*drho+f2a155*drho**2+f3a155*drho**3
      fea455= f0a455+f1a455*drho+f2a455*drho**2+f3a455*drho**3
   
      fea1111= f0a1111+f1a1111*drho+f2a1111*drho**2
      fea1112= f0a1112+f1a1112*drho+f2a1112*drho**2
      fea1114= f0a1114+f1a1114*drho+f2a1114*drho**2
      fea1122= f0a1122+f1a1122*drho+f2a1122*drho**2
      fea1123= f0a1123+f1a1123*drho+f2a1123*drho**2
      fea1124= f0a1124+f1a1124*drho+f2a1124*drho**2
      fea1125= f0a1125+f1a1125*drho+f2a1125*drho**2
      fea1144= f0a1144+f1a1144*drho+f2a1144*drho**2
      fea1155= f0a1155+f1a1155*drho+f2a1155*drho**2
      fea1244= f0a1244+f1a1244*drho+f2a1244*drho**2
      fea1255= f0a1255+f1a1255*drho+f2a1255*drho**2
      fea1444= f0a1444+f1a1444*drho+f2a1444*drho**2
      fea1455= f0a1455+f1a1455*drho+f2a1455*drho**2
      fea4444= f0a4444+f1a4444*drho+f2a4444*drho**2
   
      if (parmax>112) then 
        !
        fea44444 = f0a44444  + f1a44444 *drho+ f2a44444 *drho**2
        fea33455 = f0a33455  + f1a33455 *drho+ f2a33455 *drho**2
        fea33445 = f0a33445  + f1a33445 *drho+ f2a33445 *drho**2
        fea33345 = f0a33345  + f1a33345 *drho+ f2a33345 *drho**2
        fea33344 = f0a33344  + f1a33344 *drho+ f2a33344 *drho**2
        fea33334 = f0a33334  + f1a33334 *drho+ f2a33334 *drho**2
        fea33333 = f0a33333  + f1a33333 *drho+ f2a33333 *drho**2
        fea25555 = f0a25555  + f1a25555 *drho+ f2a25555 *drho**2
        fea24455 = f0a24455  + f1a24455 *drho+ f2a24455 *drho**2
        fea24445 = f0a24445  + f1a24445 *drho+ f2a24445 *drho**2
        fea23333 = f0a23333  + f1a23333 *drho+ f2a23333 *drho**2
        fea13455 = f0a13455  + f1a13455 *drho+ f2a13455 *drho**2
        fea13445 = f0a13445  + f1a13445 *drho+ f2a13445 *drho**2
        fea13345 = f0a13345  + f1a13345 *drho+ f2a13345 *drho**2
        fea12355 = f0a12355  + f1a12355 *drho+ f2a12355 *drho**2
        fea11334 = f0a11334  + f1a11334 *drho+ f2a11334 *drho**2
        fea11333 = f0a11333  + f1a11333 *drho+ f2a11333 *drho**2
        fea11255 = f0a11255  + f1a11255 *drho+ f2a11255 *drho**2
        fea11245 = f0a11245  + f1a11245 *drho+ f2a11245 *drho**2
        fea11234 = f0a11234  + f1a11234 *drho+ f2a11234 *drho**2
        fea11233 = f0a11233  + f1a11233 *drho+ f2a11233 *drho**2
        fea11135 = f0a11135  + f1a11135 *drho+ f2a11135 *drho**2
        fea11134 = f0a11134  + f1a11134 *drho+ f2a11134 *drho**2
        fea11123 = f0a11123  + f1a11123 *drho+ f2a11123 *drho**2
        fea555555= f0a555555 + f1a555555*drho+ f2a555555*drho**2
        fea444444= f0a444444 + f1a444444*drho+ f2a444444*drho**2
        fea335555= f0a335555 + f1a335555*drho+ f2a335555*drho**2
        fea334455= f0a334455 + f1a334455*drho+ f2a334455*drho**2
        fea334445= f0a334445 + f1a334445*drho+ f2a334445*drho**2
        fea333555= f0a333555 + f1a333555*drho+ f2a333555*drho**2
        fea333333= f0a333333 + f1a333333*drho+ f2a333333*drho**2
        fea244555= f0a244555 + f1a244555*drho+ f2a244555*drho**2
        fea244455= f0a244455 + f1a244455*drho+ f2a244455*drho**2
        fea233445= f0a233445 + f1a233445*drho+ f2a233445*drho**2
        fea233444= f0a233444 + f1a233444*drho+ f2a233444*drho**2
        fea233345= f0a233345 + f1a233345*drho+ f2a233345*drho**2
        fea233344= f0a233344 + f1a233344*drho+ f2a233344*drho**2
        fea233335= f0a233335 + f1a233335*drho+ f2a233335*drho**2
        fea223355= f0a223355 + f1a223355*drho+ f2a223355*drho**2
        fea222335= f0a222335 + f1a222335*drho+ f2a222335*drho**2
        fea222334= f0a222334 + f1a222334*drho+ f2a222334*drho**2
        fea222333= f0a222333 + f1a222333*drho+ f2a222333*drho**2
        fea222255= f0a222255 + f1a222255*drho+ f2a222255*drho**2
        fea222245= f0a222245 + f1a222245*drho+ f2a222245*drho**2
        fea222233= f0a222233 + f1a222233*drho+ f2a222233*drho**2
        fea222224= f0a222224 + f1a222224*drho+ f2a222224*drho**2
        fea145555= f0a145555 + f1a145555*drho+ f2a145555*drho**2
        fea134444= f0a134444 + f1a134444*drho+ f2a134444*drho**2
        fea133444= f0a133444 + f1a133444*drho+ f2a133444*drho**2
        fea133345= f0a133345 + f1a133345*drho+ f2a133345*drho**2
        fea133334= f0a133334 + f1a133334*drho+ f2a133334*drho**2
        fea133333= f0a133333 + f1a133333*drho+ f2a133333*drho**2
        fea124555= f0a124555 + f1a124555*drho+ f2a124555*drho**2
        fea124455= f0a124455 + f1a124455*drho+ f2a124455*drho**2
        fea123455= f0a123455 + f1a123455*drho+ f2a123455*drho**2
        fea123345= f0a123345 + f1a123345*drho+ f2a123345*drho**2
        fea113555= f0a113555 + f1a113555*drho+ f2a113555*drho**2
        fea113345= f0a113345 + f1a113345*drho+ f2a113345*drho**2
        fea112355= f0a112355 + f1a112355*drho+ f2a112355*drho**2
        fea112335= f0a112335 + f1a112335*drho+ f2a112335*drho**2
        fea112233= f0a112233 + f1a112233*drho+ f2a112233*drho**2
        fea111444= f0a111444 + f1a111444*drho+ f2a111444*drho**2
        fea111234= f0a111234 + f1a111234*drho+ f2a111234*drho**2
        fea111233= f0a111233 + f1a111233*drho+ f2a111233*drho**2
        fea111123= f0a111123 + f1a111123*drho+ f2a111123*drho**2
       !
      endif 
   
      v1 = (y3+y2+y1)*fea1
   
      v2 = (y2*y3+y1*y3+y1*y2)*fea12                                                                 &
       +(y2**2+y3**2+y1**2)*fea11                                                                    &
       +(-sqrt(3.0d0)*y3*y5/2.0d0-y3*y4/2.0d0+y1*y4+sqrt(3.0d0)*y2*y5/2.0d0-y2*y4/2.0d0)*fea14 &
       +(y5**2+y4**2)*fea44
   
      v3 = (y1*y3*y4+y1*y2*y4-2.0d0*y2*y3*y4+sqrt(3.0d0)*y1*y2*y5-sqrt(3.0d0)*y1*y3*y5)*fea124   &
       +(3.0d0/4.0d0*y3*y4**2-sqrt(3.0d0)*y3*y4*y5/2.0d0+y1*y5**2+y2*y5**2/4.0d0               & 
       +3.0d0/4.0d0*y2*y4**2+sqrt(3.0d0)*y2*y4*y5/2.0d0+y3*y5**2/4.0d0)*fea155                 &
       +(y2*y3**2+y1*y3**2+y1**2*y3+y1*y2**2+y2**2*y3+y1**2*y2)*fea112+                             &
       (-y4**3/3.0d0+y4*y5**2)*fea455+fea123*y1*y2*y3                                              &
       +(y1*y4**2+3.0d0/4.0d0*y3*y5**2+3.0d0/4.0d0*y2*y5**2+y2*y4**2/4.0d0                     &
       -sqrt(3.0d0)*y2*y4*y5/2.0d0+sqrt(3.0d0)*y3*y4*y5/2.0d0+y3*y4**2/4.0d0)*fea144           &
       +(y3**3+y2**3+y1**3)*fea111+(-y2**2*y4/2.0d0-y3**2*y4/2.0d0+sqrt(3.0d0)*y2**2*y5/2.0d0   & 
       +y1**2*y4-sqrt(3.0d0)*y3**2*y5/2.0d0)*fea114
       !
      s2 = (y4**4+y5**4+2.0d0*y4**2*y5**2)*fea4444+(3.0d0/8.0d0*sqrt(3.0d0)*&
       y2*y5**3-3.0d0/8.0d0*sqrt(3.0d0)*y3*y4**2*y5-3.0d0/8.0d0*sqrt(3.0d0)*y3*&
       y5**3-9.0d0/8.0d0*y2*y4*y5**2-y3*y4**3/8.0d0-y2*y4**3/8.0d0-9.0d0/8.0d0*&
       y3*y4*y5**2+y1*y4**3+3.0d0/8.0d0*sqrt(3.0d0)*y2*y4**2*y5)*fea1444 &
       +(3.0d0/4.0d0*y2**2*y4**2+3.0d0/4.0d0*y3**2*y4**2+y1**2*y5**2+y3**2*y5**2/4.0d0 &
       -sqrt(3.0d0)*y3**2*y4*y5/2.0d0+sqrt(3.0d0)*y2**2*y4*y5/2.0d0+y2**2&
       *y5**2/4.0d0)*fea1155 
       s1 = s2+(y3**2*y4**2/4.0d0+3.0d0/4.0d0*y3**2*y5**2+y1**2*y4**2+y2**2*&
       y4**2/4.0d0+sqrt(3.0d0)*y3**2*y4*y5/2.0d0-sqrt(3.0d0)*y2**2*y4*y5/2.0d0&
       +3.0d0/4.0d0*y2**2*y5**2)*fea1144+(y1**3*y4+sqrt(3.0d0)*y2**3*y5/2.0d0&
       -sqrt(3.0d0)*y3**3*y5/2.0d0-y2**3*y4/2.0d0-y3**3*y4/2.0d0)*fea1114&
       +(y2**4+y1**4+y3**4)*fea1111+(sqrt(3.0d0)*y1*y3*y4*y5+3.0d0/2.0d0*y2*y3*y5**2&
       -y2*y3*y4**2/2.0d0+y1*y2*y4**2-sqrt(3.0d0)*y1*y2*y4*y5+y1*y3*y4**2)*fea1244 
       !
      s2 = s1+(y1*y3*y5**2+y1*y2*y5**2-sqrt(3.0d0)*y1*y3*y4*y5-y2*y3*y5**&
       2/2.0d0+3.0d0/2.0d0*y2*y3*y4**2+sqrt(3.0d0)*y1*y2*y4*y5)*fea1255+&
       (-y1*y3**2*y4/2.0d0+y1**2*y3*y4-sqrt(3.0d0)*y1*y3**2*y5/2.0d0-sqrt(3.0d0)*y2&
       *y3**2*y5/2.0d0+y1**2*y2*y4+sqrt(3.0d0)*y2**2*y3*y5/2.0d0-y2**2*y3*y4&
       /2.0d0+sqrt(3.0d0)*y1*y2**2*y5/2.0d0-y2*y3**2*y4/2.0d0-y1*y2**2*y4/2.0d0&
       )*fea1124+(y1**2*y2*y5+sqrt(3.0d0)*y1*y3**2*y4/2.0d0+sqrt(3.0d0)*y1*&
       y2**2*y4/2.0d0-sqrt(3.0d0)*y2*y3**2*y4/2.0d0-sqrt(3.0d0)*y2**2*y3*y4/2.0d0&
       -y2**2*y3*y5/2.0d0+y2*y3**2*y5/2.0d0-y1*y3**2*y5/2.0d0+y1*y2**2*y5&
       /2.0d0-y1**2*y3*y5)*fea1125 
       !
      v4 = s2+(y2*y3**3+y1**3*y3+y1**3*y2+y1*y2**3+y1*y3**3+y2**3*y3)*fea1112+&
       (y2**2*y3**2+y1**2*y3**2+y1**2*y2**2)*fea1122+(y1*y2**2*y3&
       +y1**2*y2*y3+y1*y2*y3**2)*fea1123+(5.0d0/8.0d0*y2*y4*y5**2+sqrt(3.0d0)*&
       y2*y5**3/8.0d0-sqrt(3.0d0)*y3*y4**2*y5/8.0d0+sqrt(3.0d0)*y2*y4**2*y5/8.0d0&
       -3.0d0/8.0d0*y2*y4**3+y1*y4*y5**2-sqrt(3.0d0)*y3*y5**3/8.0d0&
       +5.0d0/8.0d0*y3*y4*y5**2-3.0d0/8.0d0*y3*y4**3)*fea1455
   
      v5 = 0 ;  v6 = 0 
      !
      if (parmax>112) then 
         !   
         s3 = (y4**5-2.0d0*y4**3*y5**2-3.0d0*y4*y5**4)*fea44444+(-4.0d0*y3*y4*&
         y5**3*sqrt(3.0d0)+9.0d0*y1*y4**2*y5**2-3.0d0/2.0d0*y1*y4**4+4.0d0*y2*y4&
         *y5**3*sqrt(3.0d0)+3.0d0*y2*y4**4+5.0d0/2.0d0*y1*y5**4+3.0d0*y3*y4**4+&
         y2*y5**4+y3*y5**4)*fea25555+(-y2*y4**4+y3*y4**2*y5**2-2.0d0*y2*y4*y5&
         **3*sqrt(3.0d0)-y3*y4**4-7.0d0/2.0d0*y1*y4**2*y5**2-3.0d0/4.0d0*y1*y5**4&
         +2.0d0*y3*y4*y5**3*sqrt(3.0d0)+y2*y4**2*y5**2+5.0d0/4.0d0*y1*y4**4)*fea24455 
         !
         s2 = s3+(y2*y4**3*y5-3.0d0*y3*y4*y5**3+2.0d0/3.0d0*y3*y4**4*sqrt(3.0d0&
         )+3.0d0/4.0d0*y1*y5**4*sqrt(3.0d0)+3.0d0*y2*y4*y5**3-&
         7.0d0/12.0d0*y1*y4**4*sqrt(3.0d0)+3.0d0/2.0d0*y1*y4**2*y5**2*sqrt(3.0d0)-y3*y4**3*y5&
         +2.0d0/3.0d0*y2*y4**4*sqrt(3.0d0))*fea24445+(-y2**2*y5**3+y3**2*y4**2*y5+ &
         y3**2*y5**3+4.0d0/9.0d0*y2**2*y4**3*sqrt(3.0d0)-5.0d0/9.0d0*y1**2*y4**3*&
         sqrt(3.0d0)+4.0d0/9.0d0*y3**2*y4**3*sqrt(3.0d0)-y2**2*y4**2*y5&
         -y1**2*y4*y5**2*sqrt(3.0d0))*fea33445+(y3**2*y4*y5**2-y1**2*y4**3/3.0d0&
         -y3**2*y4**3/3.0d0+y1**2*y4*y5**2+y2**2*y4*y5**2-y2**2*y4**3/3.0d0)*fea33455 
         !
         s1 = s2+(-y2**3*y4*y5+y3**3*y4*y5+y2**3*y5**2*sqrt(3.0d0)/3.0d0+y1**&
         3*y4**2*sqrt(3.0d0)/2.0d0+y3**3*y5**2*sqrt(3.0d0)/3.0d0- & 
         y1**3*y5**2*sqrt(3.0d0)/6.0d0)*fea33345+(y3**3*y4**2+y3**3*y5**2+y2**3*y4**2+y2**3&
         *y5**2+y1**3*y5**2+y1**3*y4**2)*fea33344+(y3**4*y4+sqrt(3.0d0)*y3**&
         4*y5+y2**4*y4-2.0d0*y1**4*y4-sqrt(3.0d0)*y2**4*y5)*fea33334+(y2**5+ &
         y3**5+y1**5)*fea33333+(-4.0d0/9.0d0*y1*y2*y4**3*sqrt(3.0d0)-y1*y2*y5**3+ &
         y1*y3*y4**2*y5+y2*y3*y4*y5**2*sqrt(3.0d0)-y1*y2*y4**2*y5+5.0d0/9.0d0&
         *y2*y3*y4**3*sqrt(3.0d0)-4.0d0/9.0d0*y1*y3*y4**3*sqrt(3.0d0)+y1*y3*y5&
         **3)*fea13445+(y2*y3*y4*y5**2+y1*y2*y4*y5**2-y2*y3*y4**3/3.0d0- & 
         y1*y2*y4**3/3.0d0-y1*y3*y4**3/3.0d0+y1*y3*y4*y5**2)*fea13455 
         
         s3 = s1+(y1**2*y3*y5**2+y2**2*y3*y4**2+y2**2*y3*y5**2+y1*y2**2*y5**2+&
         y1**2*y2*y5**2+y1*y2**2*y4**2+y2*y3**2*y4**2+y1*y3**2*y4**2+&
         y1**2*y3*y4**2+y1**2*y2*y4**2+y1*y3**2*y5**2+y2*y3**2*y5**2)*fea11255&
         +(2.0d0/3.0d0*y1**2*y3*y4**2*sqrt(3.0d0)+y1*y3**2*y5**2*sqrt(3.0d0)/2.0d0+&
         y1*y2**2*y5**2*sqrt(3.0d0)/2.0d0+y2**2*y3*y5**2*sqrt(3.0d0)/2.0d0-&
         y1*y2**2*y4*y5+y2*y3**2*y4*y5+y1*y3**2*y4*y5-y2**2*y3*y4*y5+y2*y3**&
         2*y4**2*sqrt(3.0d0)/6.0d0+y1*y3**2*y4**2*sqrt(3.0d0)/6.0d0+y1*y2**2*y4&
         **2*sqrt(3.0d0)/6.0d0+2.0d0/3.0d0*y1**2*y2*y4**2*sqrt(3.0d0)+&
         y2*y3**2*y5**2*sqrt(3.0d0)/2.0d0+y2**2*y3*y4**2*sqrt(3.0d0)/6.0d0)*fea13345 
         s4 = s3+(y1**2*y2*y4*y5+y1**2*y3*y4**2*sqrt(3.0d0)/3.0d0+y1**2*y2*y4&
         **2*sqrt(3.0d0)/3.0d0-y1*y2**2*y4**2*sqrt(3.0d0)/6.0d0+y2*y3**2*y4*y5-&
         y2**2*y3*y4*y5-y1**2*y3*y4*y5+y2*y3**2*y4**2*sqrt(3.0d0)/3.0d0+y1*y2&
         **2*y5**2*sqrt(3.0d0)/2.0d0-y1*y3**2*y4**2*sqrt(3.0d0)/6.0d0+y2**2*y3*&
         y4**2*sqrt(3.0d0)/3.0d0+y1*y3**2*y5**2*sqrt(3.0d0)/2.0d0)*fea11245 
         s2 = s4+(-y1**3*y2*y5+y1**3*y3*y5+y2**3*y3*y5/2.0d0-y1*y2**3*y4*sqrt(3.0d0)/2.0d0-&
         y1*y2**3*y5/2.0d0-y2*y3**3*y5/2.0d0+y1*y3**3*y5/2.0d0+y2&
         **3*y3*y4*sqrt(3.0d0)/2.0d0+y2*y3**3*y4*sqrt(3.0d0)/2.0d0-y1*y3**3*y4*&
         sqrt(3.0d0)/2.0d0)*fea11135+(y1**3*y3*y4-y2**3*y3*y4/2.0d0+y1**3*y2*y4-&
         y2*y3**3*y4/2.0d0-y1*y3**3*y4/2.0d0+y1*y2**3*y5*sqrt(3.0d0)/2.0d0+y2&
         **3*y3*y5*sqrt(3.0d0)/2.0d0-y2*y3**3*y5*sqrt(3.0d0)/2.0d0-y1*y2**3*y4/&
         2.0d0-y1*y3**3*y5*sqrt(3.0d0)/2.0d0)*fea11134 
         
         v5 = s2+(y1*y2**4+y1**4*y3+y1**4*y2+y2**4*y3+y2*y3**4+y1*y3**4)*fea23333+&
         (-2.0d0*y2**2*y3**2*y4+y1**2*y2**2*y4-sqrt(3.0d0)*y1**2*y3**2&
         *y5+sqrt(3.0d0)*y1**2*y2**2*y5+y1**2*y3**2*y4)*fea11334+(y1**2*y3**&
         3+y1**3*y3**2+y2**2*y3**3+y1**2*y2**3+y1**3*y2**2+y2**3*y3**2)*fea11333+&
         (y1*y2*y3*y4**2+y1*y2*y3*y5**2)*fea12355+(-y1*y2*y3**2*y4/2.0d0-&
         y1*y2**2*y3*y4/2.0d0-sqrt(3.0d0)*y1*y2*y3**2*y5/2.0d0+y1**2*y2*y3*y4+&
         sqrt(3.0d0)*y1*y2**2*y3*y5/2.0d0)*fea11234+(y1*y2**3*y3+y1*y2*y3**3+&
         y1**3*y2*y3)*fea11123+(y1**2*y2**2*y3+y1*y2**2*y3**2+y1**2*y2*y3**2)*fea11233
         
         s3 = (y2**3*y4**3*sqrt(3.0d0)-y2**3*y4**2*y5+y3**3*y4**2*y5-&
         5.0d0/3.0d0*y2**3*y4*y5**2*sqrt(3.0d0)+y3**3*y4**3*sqrt(3.0d0)-5.0d0/3.0d0*y3**&
         3*y4*y5**2*sqrt(3.0d0)-y2**3*y5**3+y3**3*y5**3-8.0d0/3.0d0*y1**3*y4*y5**2*sqrt(3.0d0))*fea333555+&
         (y1**4*y5**2*sqrt(3.0d0)/2.0d0+y2**4*y4*y5+y2**4*y4**2*sqrt(3.0d0)/3.0d0+&
         y3**4*y4**2*sqrt(3.0d0)/3.0d0-y3**4*y4&
         *y5-y1**4*y4**2*sqrt(3.0d0)/6.0d0)*fea222245+(y1*y3**5+y1*y2**5+y2**&
         5*y3+y1**5*y3+y1**5*y2+y2*y3**5)*fea133333+(y1**4*y3*y4-2.0d0*y2**4&
         *y3*y4+y1**4*y2*y4+y1*y2**4*y5*sqrt(3.0d0)+y1*y3**4*y4-2.0d0*y2*y3**&
         4*y4+y1**4*y2*y5*sqrt(3.0d0)-y1*y3**4*y5*sqrt(3.0d0)-y1**4*y3*y5*sqrt(3.0d0)+&
         y1*y2**4*y4)*fea133334+(-y1*y2*y3*y4**3/3.0d0+y1*y2*y3*y4*y5**2)*fea123455 
         
         s4 = s3+(2.0d0/3.0d0*sqrt(3.0d0)*y1*y2**2*y3**2*y4-y1**2*y2**2*y3*y5-&
         sqrt(3.0d0)*y1**2*y2**2*y3*y4/3.0d0+y1**2*y2*y3**2*y5-&
         sqrt(3.0d0)*y1**2*y2*y3**2*y4/3.0d0)*fea112335+(y1*y2**2*y3*y5**2+y1*y2*y3**2*y5**2+&
         y1*y2*y3**2*y4**2+y1*y2**2*y3*y4**2+y1**2*y2*y3*y4**2+y1**2*y2*y3*y5**2)*fea112355 
         
         s2 = s4+(y2**3*y3**2*y5-y1**3*y2**2*y5/2.0d0-y1**2*y3**3*y5/2.0d0-y2&
         **2*y3**3*y5+y1**3*y2**2*y4*sqrt(3.0d0)/2.0d0-y1**2*y2**3*y4*sqrt(3.0d0)/2.0d0+&
         y1**3*y3**2*y5/2.0d0+y1**2*y2**3*y5/2.0d0+y1**3*y3**2*y4*sqrt(3.0d0)/2.0d0-&
         y1**2*y3**3*y4*sqrt(3.0d0)/2.0d0)*fea222335+(-y1**2*y2&
         **2*y5**2*sqrt(3.0d0)/2.0d0-y1**2*y3**2*y5**2*sqrt(3.0d0)/2.0d0-y1**2*&
         y2**2*y4**2*sqrt(3.0d0)/6.0d0-y1**2*y2**2*y4*y5-2.0d0/3.0d0*y2**2*y3**&
         2*y4**2*sqrt(3.0d0)+y1**2*y3**2*y4*y5-y1**2*y3**2*y4**2*sqrt(3.0d0)/&
         6.0d0)*fea113345+(y2**2*y3**2*y5**2+y2**2*y3**2*y4**2+y1**2*y2**2*y5**2+&
         y1**2*y3**2*y4**2+y1**2*y3**2*y5**2+y1**2*y2**2*y4**2)*fea223355 
         
         s3 = s2+(y1*y2*y3**2*y4**2*sqrt(3.0d0)/6.0d0+y1*y2*y3**2*y4*y5+y1*y2&
         *y3**2*y5**2*sqrt(3.0d0)/2.0d0+2.0d0/3.0d0*y1**2*y2*y3*y4**2*sqrt(3.0d0&
         )-y1*y2**2*y3*y4*y5+y1*y2**2*y3*y4**2*sqrt(3.0d0)/6.0d0+y1*y2**2*y3*&
         y5**2*sqrt(3.0d0)/2.0d0)*fea123345+(-y1**3*y2**2*y5*sqrt(3.0d0)/2.0d0-&
         y1**3*y2**2*y4/2.0d0-y1**3*y3**2*y4/2.0d0-y1**2*y2**3*y4/2.0d0+y1**3*&
         y3**2*y5*sqrt(3.0d0)/2.0d0-y1**2*y3**3*y4/2.0d0+y2**3*y3**2*y4-y1**2*&
         y2**3*y5*sqrt(3.0d0)/2.0d0+y2**2*y3**3*y4+y1**2*y3**3*y5*sqrt(3.0d0)/&
         2.0d0)*fea222334+(3.0d0*y3**2*y4**4+5.0d0/2.0d0*y1**2*y5**4+y2**2*y5**&
         4+3.0d0*y2**2*y4**4-4.0d0*y3**2*y4*y5**3*sqrt(3.0d0)+y3**2*y5**4+9.0d0&
         *y1**2*y4**2*y5**2-3.0d0/2.0d0*y1**2*y4**4+4.0d0*y2**2*y4*y5**3*sqrt(&
         3.0d0))*fea335555+(y1**3*y2**3+y1**3*y3**3+y2**3*y3**3)*fea222333 
         
         s4 = s3+(y3*y4**5/5.0d0-y2*y4**4*y5*sqrt(3.0d0)/2.0d0-2.0d0/5.0d0*y1*y4&
         **5-2.0d0*y1*y4**3*y5**2-3.0d0/10.0d0*y2*y5**5*sqrt(3.0d0)+y3*y4**3*y5&
         **2+y3*y4**4*y5*sqrt(3.0d0)/2.0d0+y2*y4**3*y5**2+3.0d0/10.0d0*y3*y5**5&
         *sqrt(3.0d0)+y2*y4**5/5.0d0)*fea244455+(y2**5*y4-2.0d0*y1**5*y4-sqrt(&
         3.0d0)*y2**5*y5+y3**5*y4+sqrt(3.0d0)*y3**5*y5)*fea222224 
         
         s5 = s4+(-y3*y5**5*sqrt(3.0d0)/5.0d0+y2*y5**5*sqrt(3.0d0)/5.0d0+y1*y4*&
         y5**4-7.0d0/15.0d0*y2*y4**5+y2*y4**4*y5*sqrt(3.0d0)/3.0d0-y3*y4**4*y5*&
         sqrt(3.0d0)/3.0d0+y3*y4*y5**4+y2*y4*y5**4+2.0d0*y1*y4**3*y5**2-7.0d0/15.0d0*y3*y4**5-&
         y1*y4**5/15.0d0)*fea145555 
         
         s1 = s5+(-sqrt(3.0d0)*y1*y2*y3**3*y5/2.0d0+y1**3*y2*y3*y4+sqrt(3.0d0)&
         *y1*y2**3*y3*y5/2.0d0-y1*y2**3*y3*y4/2.0d0-y1*y2*y3**3*y4/2.0d0)*fea111234+&
         (y3*y4**4*y5/3.0d0+y3*y4**5*sqrt(3.0d0)/18.0d0-y2*y4**4*y5/3.0d0&
         -y2*y4*y5**4*sqrt(3.0d0)/2.0d0-y3*y4**2*y5**3+2.0d0/9.0d0*y1*y4**5*sqrt(3.0d0)+&
         y2*y4**5*sqrt(3.0d0)/18.0d0+y2*y4**2*y5**3-2.0d0/3.0d0*y1*y4**&
         3*y5**2*sqrt(3.0d0)-y3*y4*y5**4*sqrt(3.0d0)/2.0d0)*fea244555+(y1*y2*y4**2*y5**2-&
         3.0d0/4.0d0*y2*y3*y4**4-y1*y2*y5**4-y1*y3*y5**4+5.0d0/4.0d0*y2*y3*y5**4+&
         y1*y3*y4**2*y5**2-7.0d0/2.0d0*y2*y3*y4**2*y5**2-2.0d0*y1&
         *y2*y4**3*y5*sqrt(3.0d0)+2.0d0*y1*y3*y4**3*y5*sqrt(3.0d0))*fea124455 
         
         s3 = s1+(y2**6+y1**6+y3**6)*fea333333+(y1*y2**4*y3+y1**4*y2*y3+y1*&
         y2*y3**4)*fea111123+fea112233*y1**2*y2**2*y3**2+(y1**4*y4**2+y2**4&
         *y4**2+y2**4*y5**2+y3**4*y4**2+y1**4*y5**2+y3**4*y5**2)*fea222255 
         s4 = s3+(3.0d0*y1*y3*y5**4+y1*y3*y4**4+9.0d0*y2*y3*y4**2*y5**2-3.0d0/&
         2.0d0*y2*y3*y5**4-4.0d0*y1*y3*y4**3*y5*sqrt(3.0d0)+y1*y2*y4**4+&
         4.0d0*y1*y2*y4**3*y5*sqrt(3.0d0)+3.0d0*y1*y2*y5**4+5.0d0/2.0d0*y2*y3*y4**4)*fea134444+&
         (-y1*y3**2*y5**3*sqrt(3.0d0)/3.0d0-7.0d0/3.0d0*y1**2*y3*y4*y5&
         **2+5.0d0/3.0d0*y1*y2**2*y4**2*y5*sqrt(3.0d0)-13.0d0/3.0d0*y2**2*y3*y4*&
         y5**2-4.0d0/3.0d0*y2*y3**2*y5**3*sqrt(3.0d0)-7.0d0/3.0d0*y1**2*y2*y4*y5&
         **2-16.0d0/3.0d0*y1*y3**2*y4*y5**2+4.0d0/3.0d0*y1**2*y3*y4**2*y5*sqrt(&
         3.0d0)+4.0d0/3.0d0*y2**2*y3*y5**3*sqrt(3.0d0)+3.0d0*y1**2*y2*y4**3+&
         y2*y3**2*y4**3+y1*y2**2*y5**3*sqrt(3.0d0)/3.0d0+y2**2*y3*y4**3-13.0d0/3.0d0&
         *y2*y3**2*y4*y5**2-5.0d0/3.0d0*y1*y3**2*y4**2*y5*sqrt(3.0d0)-&
         4.0d0/3.0d0*y1**2*y2*y4**2*y5*sqrt(3.0d0)+3.0d0*y1**2*y3*y4**3-16.0d0/3.0d0*y1*&
         y2**2*y4*y5**2)*fea233444 
         
         s5 = s4+(2.0d0*y1*y3**2*y5**3+4.0d0*y2*y3**2*y5**3+4.0d0*y2**2*y3*y4*&
         y5**2*sqrt(3.0d0)-2.0d0*y1*y2**2*y5**3+y1**2*y3*y4*y5**2*sqrt(3.0d0)+&
         6.0d0*y1*y3**2*y4**2*y5-6.0d0*y1*y2**2*y4**2*y5-3.0d0*y1**2*y3*y4**2*&
         y5+y1**2*y2*y4*y5**2*sqrt(3.0d0)+4.0d0*y1*y3**2*y4*y5**2*sqrt(3.0d0)-&
         3.0d0*y1**2*y2*y4**3*sqrt(3.0d0)-4.0d0*y2**2*y3*y5**3+3.0d0*y1**2*y2*y4**2*y5-&
         y1**2*y2*y5**3+y1**2*y3*y5**3-3.0d0*y1**2*y3*y4**3*sqrt(3.0d0&
         )+4.0d0*y2*y3**2*y4*y5**2*sqrt(3.0d0)+4.0d0*y1*y2**2*y4*y5**2*sqrt(3.0d0))*fea113555 
         
         s2 = s5+(-2.0d0/3.0d0*y3**2*y4**4*sqrt(3.0d0)-3.0d0/2.0d0*y1**2*y4**2*y5**2*sqrt(3.0d0)-&
         3.0d0/4.0d0*y1**2*y5**4*sqrt(3.0d0)-y2**2*y4**3*y5+&
         7.0d0/12.0d0*y1**2*y4**4*sqrt(3.0d0)+y3**2*y4**3*y5+3.0d0*y3**2*y4*y5**3&
         -2.0d0/3.0d0*y2**2*y4**4*sqrt(3.0d0)-3.0d0*y2**2*y4*y5**3)*fea334445+(&
         -3.0d0*y1*y3*y4**3*y5+2.0d0/3.0d0*y1*y2*y5**4*sqrt(3.0d0)-y1*y3*y4*y5**3+&
         2.0d0/3.0d0*y1*y3*y5**4*sqrt(3.0d0)+3.0d0*y1*y2*y4**3*y5-7.0d0/12.0d0&
         *y2*y3*y5**4*sqrt(3.0d0)+3.0d0/2.0d0*y2*y3*y4**2*y5**2*sqrt(3.0d0)+y1*&
         y2*y4*y5**3+3.0d0/4.0d0*y2*y3*y4**4*sqrt(3.0d0))*fea124555+(2.0d0*y3**&
         2*y4*y5**3*sqrt(3.0d0)-7.0d0/2.0d0*y1**2*y4**2*y5**2+y2**2*y4**2*y5**&
         2-y2**2*y4**4-y3**2*y4**4-2.0d0*y2**2*y4*y5**3*sqrt(3.0d0)-3.0d0/4.0d0&
         *y1**2*y5**4+5.0d0/4.0d0*y1**2*y4**4+y3**2*y4**2*y5**2)*fea334455 
         s3 = s2+(-6.0d0*y4**2*y5**4+9.0d0*y4**4*y5**2+y5**6)*fea555555+(y2*y3**3*y4**2+&
         y2*y3**3*y5**2+y1*y3**3*y4**2+y1*y2**3*y4**2+y1**3*y2*y4**2+&
         y1*y2**3*y5**2+y1**3*y3*y5**2+y1**3*y3*y4**2+y1**3*y2*y5**2+y2**3*y3*y4**2+&
         y1*y3**3*y5**2+y2**3*y3*y5**2)*fea233344+(y1*y2**3*y5**2*sqrt(3.0d0)/6.0d0-&
         y2**3*y3*y5**2*sqrt(3.0d0)/3.0d0-y2*y3**3*y5**2&
         *sqrt(3.0d0)/3.0d0+y1**3*y2*y4*y5-y1**3*y2*y5**2*sqrt(3.0d0)/3.0d0-&
         y1**3*y3*y4*y5-y1**3*y3*y5**2*sqrt(3.0d0)/3.0d0-y1*y3**3*y4**2*sqrt(3.0d0&
         )/2.0d0+y1*y3**3*y5**2*sqrt(3.0d0)/6.0d0-y2**3*y3*y4*y5+y2*y3**3*y4*&
         y5-y1*y2**3*y4**2*sqrt(3.0d0)/2.0d0)*fea233345+(-3.0d0*y2**3*y4*y5**2&
         +y3**3*y4**3-3.0d0*y3**3*y4*y5**2-3.0d0*y1**3*y4*y5**2+y2**3*y4**3+&
         y1**3*y4**3)*fea111444+(y1*y2**3*y3**2+y1**3*y2**2*y3+y1**2*y2**3*y3+&
         y1*y2**2*y3**3+y1**2*y2*y3**3+y1**3*y2*y3**2)*fea111233 
         
         s4 = s3+(9.0d0*y4**2*y5**4-6.0d0*y4**4*y5**2+y4**6)*fea444444+(-5.0d0&
         /3.0d0*y1*y2**2*y4**2*y5*sqrt(3.0d0)+y1*y2**2*y4**3-4.0d0/3.0d0*y1**2*&
         y3*y4**2*y5*sqrt(3.0d0)-2.0d0*y1**2*y2*y4**3-y1*y2**2*y5**3*sqrt(3.0d0&
         )/3.0d0+4.0d0/3.0d0*y2**2*y3*y4*y5**2-4.0d0/3.0d0*y2**2*y3*y5**3*sqrt(&
         3.0d0)-2.0d0*y1**2*y3*y4**3+7.0d0/3.0d0*y1*y2**2*y4*y5**2-2.0d0/3.0d0*y1&
         **2*y3*y4*y5**2+y1*y3**2*y4**3+4.0d0/3.0d0*y2*y3**2*y5**3*sqrt(3.0d0)&
         +y1*y3**2*y5**3*sqrt(3.0d0)/3.0d0+4.0d0/3.0d0*y1**2*y2*y4**2*y5*sqrt(3.0d0)&
         +4.0d0/3.0d0*y2*y3**2*y4*y5**2+5.0d0/3.0d0*y1*y3**2*y4**2*y5*sqrt(&
         3.0d0)-2.0d0/3.0d0*y1**2*y2*y4*y5**2+7.0d0/3.0d0*y1*y3**2*y4*y5**2)*fea133444 
         
         s5 = s4+(-y1**3*y2*y4*y5+2.0d0/3.0d0*y2**3*y3*y5**2*sqrt(3.0d0)+y1*y3&
         **3*y4**2*sqrt(3.0d0)/2.0d0+y1**3*y3*y4**2*sqrt(3.0d0)/2.0d0+y1**3*y3*&
         y5**2*sqrt(3.0d0)/6.0d0+y1**3*y2*y5**2*sqrt(3.0d0)/6.0d0+y1**3*y3*y4*y5+&
         y1*y2**3*y5**2*sqrt(3.0d0)/6.0d0+y1**3*y2*y4**2*sqrt(3.0d0)/2.0d0+&
         2.0d0/3.0d0*y2*y3**3*y5**2*sqrt(3.0d0)-y1*y2**3*y4*y5+y1*y2**3*y4**2*sqrt(3.0d0)/2.0d0+&
         y1*y3**3*y5**2*sqrt(3.0d0)/6.0d0+y1*y3**3*y4*y5)*fea133345 
         
         v6 = s5+(-y2**2*y3*y4**2*y5+y1**2*y3*y4*y5**2*sqrt(3.0d0)/3.0d0+y2*y3**2*y4**2*y5+&
         y2*y3**2*y5**3-y1*y2**2*y5**3+4.0d0/3.0d0*y2**2*y3*y4*&
         y5**2*sqrt(3.0d0)+4.0d0/3.0d0*y2*y3**2*y4*y5**2*sqrt(3.0d0)-y1*y2**2*y4**2*y5+&
         4.0d0/3.0d0*y1*y3**2*y4*y5**2*sqrt(3.0d0)-y2**2*y3*y5**3+y1*y3**2*y5**3+&
         y1**2*y2*y4*y5**2*sqrt(3.0d0)/3.0d0-y1**2*y2*y4**3*sqrt(3.0d0)&
         +y1*y3**2*y4**2*y5-y1**2*y3*y4**3*sqrt(3.0d0)+4.0d0/3.0d0*y1*y2**&
         2*y4*y5**2*sqrt(3.0d0))*fea233445+(y2*y3**4*y4*sqrt(3.0d0)-y1**4*y2*&
         y5+y2**4*y3*y4*sqrt(3.0d0)-y1**4*y3*y4*sqrt(3.0d0)+y2*y3**4*y5-2.0d0*&
         y1*y2**4*y5+2.0d0*y1*y3**4*y5-y1**4*y2*y4*sqrt(3.0d0)+y1**4*y3*y5-y2&
         **4*y3*y5)*fea233335+(y2**2*y3**4+y1**4*y3**2+y1**2*y2**4+y2**4*y3&
         **2+y1**2*y3**4+y1**4*y2**2)*fea222233
         !
      endif
      !
      DMS_A = (v0+v1+v2+v3+v4+v5+v6)*cosrho
      !
   end function DMS_A
   
   
   
   
   
     subroutine MLlinur(dimen,npar,coeff,constant,solution,error)
   
     integer,intent(in)  :: dimen,npar
     integer,intent(out) :: error 
     real(8),intent(in)  :: coeff(npar,npar),constant(npar)
     real(8),intent(out) :: solution(npar)
     real(8)          :: a0(npar,npar)
     real(8)          :: c
     integer                   :: i1,i2,i,k8,k,l,k9
   
     !----- begin ----!
     
       do i1=1,dimen
       do i2=1,dimen 
          a0(i1,i2)=coeff(i1,i2)
       enddo
       enddo
   
       do i=1,dimen
         solution(i)=constant(i)
       enddo
       error=0
       do i=1,dimen
         c=0
         k8=i-1
         do k=1,k8
           c=c+a0(k,i)*a0(k,i)
         enddo
   
         if (c.ge.a0(i,i)) then
         !      write(6,*) '(',i,'-th adj. parameter is wrong )'
          error=i
          return
         endif
   
         a0(i,i)=sqrt(a0(i,i)-c)
         if (a0(i,i).eq.0) then
         !      write(6,*) '(',i,'-th adj. parameter is wrong )'
            error=i
            return
         endif
         k8=i+1
         do l=k8,dimen
            k9=i-1
            c=0.0
            do k=1,k9 
               c=c+a0(k,i)*a0(k,l)
            enddo
            a0(i,l)=(a0(l,i)-c)/a0(i,i)
         enddo
       enddo
       do i=1,dimen
         k8=i-1
         c=0.0
         do k=1,k8
            c=c+a0(k,i)*solution(k)
         enddo
         solution(i)=(solution(i)-c)/a0(i,i)
       enddo
       do i1=1,dimen
         i=1+dimen-i1
         k8=i+1
         c=0.0
         do k=k8,dimen
             c=c+a0(i,k)*solution(k)
         enddo
         solution(i)=(solution(i)-c)/a0(i,i)
       enddo
     return
     end subroutine MLlinur
   
   
   !
   ! vector product: v = [v1,v2]
   !
   
     subroutine vector_product(v1,v2,v)
       !
       double precision,intent(in) :: v1(3),v2(3)
       double precision :: v(3)
       !
       v(1) = v1(2)*v2(3)-v1(3)*v2(2)
       !
       v(2) = v1(3)*v2(1)-v1(1)*v2(3)
       !
       v(3) = v1(1)*v2(2)-v1(2)*v2(1)
       !
     end subroutine vector_product
   
   
   
   
