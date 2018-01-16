% Generate a JSON file containing GKW inputs/outputs with the GKDB conventions
% The standard GKW matlab scripts are used to load the data
%  
%   [out,is_ok,ok_msg]=gkw2json(flnm,proj,flpth_json,comments,N_shape,flpth_hamada) 
% 
% Inputs:
%   flnm:         scan name
%   proj:         project name
%   flpth_json:   output folder for the json files 
%   comments:     comments to be applied to all runs of the scan
%   N_shape:      order of the Fourier expansion for the plasma shape 
%   flpth_hamada: reference path for the hamada.dat files (optional)
%
% Outputs: 
%   out:          structure with the required data for the GKDB, one field per run of the scan
%   is_ok:        1 if the run was selected for the GKDB, 0 otherwise
%   ok_msg:       when is_ok=0, explains why
%
% YC - 13.04.2017
% 


function [out,is_ok,ok_msg]=gkw2json(flnm,proj,flpth_json,comments,N_shape,flpth_hamada)



% defaults
if ~exist('flnm')||isempty(flnm)
 disp('Scan name required in input')
 [out,is_ok,ok_msg]=deal([]);
 return
end

if ~exist('proj')
	proj=[];
end
flpth=gkwpath('scan',proj);

if ~exist('comments')||isempty(comments)
 comments='This entry is for testing purposes';
end

if ~exist('flpth_json')||isempty(flpth_json)
 flpth_json=gkwpath('json',proj);
end

if ~exist('N_shape')||isempty(N_shape)
 N_shape=8;
end

if ~exist('flpth_hamada')||isempty(flpth_hamada)
 flpth_hamada='/home/yann/runs/chease/JET/output/';
end

if ~(unix(['test -e ' flpth_json])==0)
 disp(['Output folder ' flpth_json ' does not exist. Please create it.'])
 [out,is_ok,ok_msg]=deal([]);
 return
end

err_th=0.003;

% load scan data
[vars,gamma,freq,pflux,eflux,vflux,s,phi,apar,G,flist,nt,bpar,Ggeom,kperp2av,kperp2av_all,Rk_av,Rk2_av,filters]=read_gkwscan(flnm,proj);

% find out if kthrho or krrho were scanned and what are their corresponding dimensions
dum=fieldnames(vars);
Ndim=length(dum);
Ikthrho=0;
Ikrrho=0;
[S,Sout]=deal(size(gamma));
for ii=1:Ndim
 switch dum{ii}
  case 'kthrho'
   Ikthrho=ii;
   Sout(ii)=1;
  case 'krrho'
   Ikrrho=ii;
   Sout(ii)=1;
 end
end

% number of output json files and corresponding index
Nout = prod(Sout);
Sdum=S.*0+1;
if Ikthrho>0 
  Sdum(Ikthrho)=S(Ikthrho);
  if Ikrrho>0
    Sdum(Ikrrho)=S(Ikrrho);
    ii_out = repmat(reshape(1:Nout,Sout),Sdum); 
  else
    ii_out = repmat(reshape(1:Nout,Sout),Sdum); 
  end
else
  if Ikrrho>0
    Sdum(Ikrrho)=S(Ikrrho);
    ii_out = repmat(reshape(1:Nout,Sout),Sdum); 
  else
    ii_out = reshape(1:Nout,Sout); 
  end
end  

is_ok_out = ones(Nout,1);
is_ok = ones(size(gamma));
ok_msg=cell(size(gamma));
out=[];

% loop over all files
for iN=1:Nout % loop over output files

 II=find(ii_out==iN);
 nsp = G{ii}.GRIDSIZE.number_of_species; % number of kinetic species
 
 % initialise arrays for fluxes and moments (with 1 eigenvalue only)
 out{iN}.particle_fluxes.phi_potential=zeros(nsp,length(II),1);
 out{iN}.particle_fluxes.a_parallel=zeros(nsp,length(II),1);
 out{iN}.particle_fluxes.b_field_parallel=zeros(nsp,length(II),1);
 out{iN}.particle_fluxes.axes={'species'  'wavevector'  'eigenvalue'};

 out{iN}.momentum_fluxes_lab.phi_potential=zeros(nsp,length(II),1);
 out{iN}.momentum_fluxes_lab.a_parallel=zeros(nsp,length(II),1);
 out{iN}.momentum_fluxes_lab.b_field_parallel=zeros(nsp,length(II),1);
 out{iN}.momentum_fluxes_lab.axes={'species'  'wavevector'  'eigenvalue'};
 out{iN}.momentum_fluxes_rotating.phi_potential=zeros(nsp,length(II),1);
 out{iN}.momentum_fluxes_rotating.a_parallel=zeros(nsp,length(II),1);
 out{iN}.momentum_fluxes_rotating.b_field_parallel=zeros(nsp,length(II),1);
 out{iN}.momentum_fluxes_rotating.axes={'species' 'wavevector'  'eigenvalue'};

 out{iN}.heat_fluxes_lab.phi_potential=zeros(nsp,length(II),1);
 out{iN}.heat_fluxes_lab.a_parallel=zeros(nsp,length(II),1);
 out{iN}.heat_fluxes_lab.b_field_parallel=zeros(nsp,length(II),1);
 out{iN}.heat_fluxes_lab.axes={'species'  'wavevector'  'eigenvalue'};
 out{iN}.heat_fluxes_rotating.phi_potential=zeros(nsp,length(II),1);
 out{iN}.heat_fluxes_rotating.a_parallel=zeros(nsp,length(II),1);
 out{iN}.heat_fluxes_rotating.b_field_parallel=zeros(nsp,length(II),1);
 out{iN}.heat_fluxes_rotating.axes={'species'  'wavevector'  'eigenvalue'};

 gkdb_moments_names={'density','parallel_velocity','perpendicular_temperature','parallel_temperature','j0_density','j0_parallel_velocity','j0_perpendicular_temperature','j0_parallel_temperature'};
 for mm=1:length(gkdb_moments_names) 
  out{iN}.moments_rotating.(['r_' gkdb_moments_names{mm}])=zeros(Ggeom{II(1)}.ns,nsp,length(II),1);
  out{iN}.moments_rotating.(['i_' gkdb_moments_names{mm}])=zeros(Ggeom{II(1)}.ns,nsp,length(II),1);
 end
 out{iN}.moments_rotating.axes={'poloidal_angle'  'species'  'wavevector'  'eigenvalue'};

 for jN=1:length(II)  % loop over wavevectors
   ii=II(jN);

 if (filters.is_conv(ii)==1)&(filters.is_parstab(ii)==1)&(filters.is_npok(ii)==1)

 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inputs 

 %%%% Geometry %%%%
 switch G{ii}.GEOM.geom_type
   case 's-alpha'
    is_ok(ii)=0;
    ok_msg{ii}='''s-alpha'' geometry not allowed in the GKDB (not a local GS equilibrium)';
    continue

   case 'circ'
    is_ok(ii)=0;
    ok_msg{ii}='''circ'' geometry not allowed in the GKDB (not a local GS equilibrium, except in special cases)';
    continue

   case 'miller'
    Rrat = 1; % Rref_GKW./Rref_GKDB
    Brat = 1; % Bref_GKW./Bref_GKDB
    dr_rat =1; % dr_GKW./dr_GKDB - not needed: GKDB definition always used in GKW
    r0=G{ii}.GEOM.eps;
    Rmil=1;
    Zmil=G{ii}.GEOM.zmil;
    k=G{ii}.GEOM.kappa;
    d=G{ii}.GEOM.delta;
    z=G{ii}.GEOM.square;
    dRmildr=G{ii}.GEOM.drmil;
    dZmildr=G{ii}.GEOM.dzmil;
    sk=G{ii}.GEOM.skappa;
    sd=G{ii}.GEOM.sdelta;
    sz=G{ii}.GEOM.ssquare;
    [R,Z] = miller2rz(r0,Rmil,Zmil,k,d,z,dRmildr,dZmildr,sk,sd,sz);
    [c,s,c_pr,s_pr,R0,Z0,err_out]=rz2genshape(R,Z,r0,N_shape,0); % shape coefficients, already normalised to Rref_GKDB
                                                                 % because R, Z and r0 are normalised to Rref_GKW=Rref_GKDB=R0
    if err_out>err_th 
      is_ok(ii)=0;
      ok_msg{ii}='Shape parametrisation inacurate, consider increasing N_shape';
      continue
    end

    if G{ii}.SPCGENERAL.betaprime_type=='ref'
      beta_pr_gkw=G{ii}.SPCGENERAL.betaprime_ref;
    else
      is_ok(ii)=0;
      ok_msg{ii}='Only betaprime_type=''ref'' is allowed for ''miller'' geometry';
      continue
    end
 
   case 'chease'
    % Read the hamada.dat file (may need to be generalised)
    dum=regexp(G{ii}.GEOM.eqfile,'(?<token>(hamada.).*(.dat))','names');
    flnm_hamada=dum.token;
    H=read_hamada(flnm_hamada,flpth_hamada);

    Rref_GKW=Ggeom{ii}.rref;
    Bref_GKW=Ggeom{ii}.bref;
    r_FS = Ggeom{ii}.eps.*Rref_GKW; % minor radius in [m]
    [c,s,c_pr,s_pr,R0,Z0,err_out]=rz2genshape(H.r,H.z,r_FS,N_shape,0); % shape coefficients, not normalised 
   
    if err_out>err_th 
      is_ok(ii)=0;
      ok_msg{ii}='Shape parametrisation inacurate, consider increasing N_shape';
      continue
    end

    Rrat = Rref_GKW./R0; % Rref_GKW./Rref_GKDB
    Brat = 1; % Bref_GKW./Bref_GKDB
    dr_rat =1; % dr_GKW./dr_GKDB - not needed: GKDB definition always used in GKW

    c=c./R0; s=s./R0;
    r0=r_FS./R0; 
    Z0=Z0./R0; R0=1; %to have the same convention for miller and chease geometry
  
    if G{ii}.SPCGENERAL.betaprime_type=='eq'
      beta_pr_gkw=Ggeom{ii}.betaprime_eq;
    else
      is_ok(ii)=0;
      ok_msg{ii}='Only betaprime_type=''eq'' is allowed for ''chease'' geometry';
      continue
    end

   otherwise
    is_ok(ii)=0;
    ok_msg{ii}=['geom_type=' G{ii}.GEOM.geom_type ' not known'];
    continue
 end

 sb_gkw = G{ii}.GEOM.signb;
 sj_gkw = G{ii}.GEOM.signj;

 out{iN}.flux_surface.r_minor = r0; 
 out{iN}.flux_surface.q       = (-sb_gkw).*(-sj_gkw).*Ggeom{ii}.q;
 out{iN}.flux_surface.magnetic_shear = Ggeom{ii}.shat;
 out{iN}.flux_surface.b_field_tor_sign = -sb_gkw;
 out{iN}.flux_surface.ip_sign = -sj_gkw;
 out{iN}.flux_surface.pressure_gradient  = -beta_pr_gkw.*Brat.^2./Rrat;
 out{iN}.flux_surface.c  = c;
 out{iN}.flux_surface.s  = s;
 out{iN}.flux_surface.dc_dr_minor  = c_pr;
 out{iN}.flux_surface.ds_dr_minor  = s_pr;


 %%%% Species parameters %%%%

 if G{ii}.SPCGENERAL.adiabatic_electrons==1
  is_ok(ii)=0;
  ok_msg{ii}='Adiabatic electrons not allowed in the GKDB';
  continue
 end
 
 Iele = find([G{ii}.SPECIES(:).z]==-1);
 if isempty(Iele)
  is_ok(ii)=0;
  ok_msg{ii}='No electron species found';
  continue
 end  
 if length(Iele)>1
  is_ok(ii)=0;
  ok_msg{ii}='Runs with multiple electron species not handled yet';
  continue
 end

 % Not used
 % me = 9.10938356e-31; % m_e [kg]
 % amu=1.660539040e-27; % atomic mass unit [kg]
 % ref_masses = [me./amu 1.007825 2.014102 3.016049 4.002602].*amu; % e, H, D, T, He [kg]
 % mD_GKDB = 3.33445e-27; % m_D [kg] in GKDB 
 % [dum,Idum]=min(abs(1-me./[me ref_masses]./G{ii}.SPECIES(Iele).mass));

 meOmD_GKDB = 2.7237e-4; % m_e/m_D in GKDB
 mrat = meOmD_GKDB./G{ii}.SPECIES(Iele).mass; %mref_GKW/mref_GKDB
 vthrat = sqrt(1./G{ii}.SPECIES(Iele).temp./mrat); % vthref_GKW/vthref_GKDB
 qrat=-1./G{ii}.SPECIES(Iele).z; %qref_GKW/qref_GKDB
 Trat=1./G{ii}.SPECIES(Iele).temp;
 nrat=1./G{ii}.SPECIES(Iele).dens;

 for jj=1:nsp % loop over species
   out{iN}.species{jj}.charge = G{ii}.SPECIES(jj).z.*qrat;
   out{iN}.species{jj}.mass = G{ii}.SPECIES(jj).mass.*mrat;
   out{iN}.species{jj}.density = G{ii}.SPECIES(jj).dens.*nrat; %correction needed if theta<>0 at s=0 + CF effects (done later in the script)
   out{iN}.species{jj}.temperature = G{ii}.SPECIES(jj).temp.*Trat;
   out{iN}.species{jj}.density_log_gradient = G{ii}.SPECIES(jj).rln./Rrat;
   out{iN}.species{jj}.temperature_log_gradient = G{ii}.SPECIES(jj).rlt./Rrat;
   out{iN}.species{jj}.toroidal_velocity_gradient = G{ii}.SPECIES(jj).uprim.*sb_gkw.*vthrat./Rrat.^2;
 end

 out{iN}.species_global.toroidal_velocity = G{ii}.ROTATION.vcor.*sb_gkw.*vthrat./Rrat;

 %%% Type of run %%%
 if strcmp(upper(G{ii}.CONTROL.method),'EXP')
  out{iN}.code.initial_value_run=1;
  nb_eiv=1;
 else
  is_ok(ii)=0;
  ok_msg{ii}='Implicit or eigenvalue runs not handled yet';
  continue
 end
 
 %%% Modes %%%% 
 out{iN}.wavevectors{jN}.poloidal_turns = G{ii}.GRIDSIZE.nperiod.*2-1;
 rhorat=mrat.*vthrat./qrat./Brat; % rhoref_GKW./rhoref_GKDB
 Imid = find(Ggeom{ii}.s_grid>-0.5 & Ggeom{ii}.s_grid<0.5 & Ggeom{ii}.r.*Rrat>1);
 gxx_th0 = interpos(Ggeom{ii}.z(Imid).*Rrat,Ggeom{ii}.g_eps_eps(Imid),Z0,0.);   % interpolation at Z=Z0 (theta=0)
 gyy_th0 = interpos(Ggeom{ii}.z(Imid).*Rrat,Ggeom{ii}.g_zeta_zeta(Imid),Z0,0.);
 gxx_s0 = interpos(Ggeom{ii}.s_grid(Imid),Ggeom{ii}.g_eps_eps(Imid),0,0.); % interpolation at s=0 (different from theta=0 for chease)
 gyy_s0 = interpos(Ggeom{ii}.s_grid(Imid),Ggeom{ii}.g_zeta_zeta(Imid),0,0.);

 if G{ii}.MODE.kr_type=='kr'|G{ii}.MODE.chin==0
  krrho =  G{ii}.MODE.krrho;
  out{iN}.wavevectors{jN}.radial_wavevector = krrho.*sqrt(gxx_th0./gxx_s0)./rhorat;
 else
  is_ok(ii)=0;
  ok_msg{ii}='Getting krrho with CHIN option not handled yet';
  continue
 end

 kthrho = G{ii}.MODE.kthrho;
 out{iN}.wavevectors{jN}.binormal_wavevector = kthrho.*sqrt(gyy_th0./gyy_s0)./rhorat;

 %%% Collisions %%%
 eV = 1.6022e-19; % elementary charge
 eps0= 8.8541878176e-12; % vacuum permittivity
 if G{ii}.CONTROL.collisions==0
  out{iN}.species_global.collisionality=0;
  out{iN}.species_global.collision_enhancement_factor=1;
  out{iN}.code.collision_ei_only=0;
  out{iN}.code.collision_pitch_only=0;
  out{iN}.code.collision_finite_larmor_radius=0;
  out{iN}.code.collision_momentum_conservation=1;
  out{iN}.code.collision_energy_conservation=1;
 else
  if G{ii}.COLLISIONS.freq_override==0
    out{iN}.species_global.collisionality = eV^4./(4*pi*eps0.^2).* G{ii}.COLLISIONS.rref .* ...
              G{ii}.COLLISIONS.nref.*G{ii}.SPECIES(Iele).dens.*1e19 ./ (1e3.*G{ii}.COLLISIONS.tref.*G{ii}.SPECIES(Iele).temp).^2; 
  else
    is_ok(ii)=0;
    ok_msg{ii}='freq_override option in COLLISIONS not treated yet';
    continue
  end

  if G{ii}.COLLISIONS.zeff>1
    % look for the value applied for Zeff in the out file produced by GKW
    % would be better to have it as proper GKW output..
    str_to_find = 'Electron/main-ion\ collision\ frequency\ scattering\ scaled\ up\ by\ Zeff='; 
    flpth_out = gkwpath('out',proj);
    [dum,dum_str]=unix(['grep ' str_to_find ' ' flpth_out flist{ii}]);
    dum=regexp(dum_str,['(' str_to_find ')(?<token>.*)'],'names');
    if isempty(dum)||(str2num(dum.token)<1 | str2num(dum.token)>1e6)
      is_ok(ii)=0;
      ok_msg{ii}='Problem when computing the scaling factor for e-i collisions';
      continue
    else
      out{iN}.species_global.collisions_enhancement_factor = str2num(dum.token);     
    end
  else
    out{iN}.species_global.collisions_enhancement_factor = 1;
  end

  out{iN}.code.collision_ei_only= (G{ii}.COLLISIONS.lorentz==1); % Lorentz also implies pitch angle only
  out{iN}.code.collision_pitch_only=(G{ii}.COLLISIONS.pitch_angle==1) & (G{ii}.COLLISIONS.en_scatter~=1 & G{ii}.COLLISIONS.friction_coll~=1);
  out{iN}.code.collision_finite_larmor_radius=0;
  out{iN}.code.collision_momentum_conservation=(G{ii}.COLLISIONS.mom_conservation==1);
  out{iN}.code.collision_energy_conservation=(G{ii}.COLLISIONS.ene_conservation==1);

 end

 %%% Debye length %%

 out{iN}.species_global.debye_length = 0; % not taken into account in GKW

 %%% EM effects %%%

 out{iN}.code.include_a_parallel = (G{ii}.CONTROL.nlapar);
 out{iN}.code.include_b_field_parallel = (G{ii}.CONTROL.nlbpar);

 if G{ii}.CONTROL.nlapar==1 | G{ii}.CONTROL.nlbpar==1
   if G{ii}.SPCGENERAL.beta_type=='ref'
     out{iN}.species_global.beta = G{ii}.SPCGENERAL.beta_ref.*Brat.^2.*G{ii}.SPECIES(Iele).dens.*G{ii}.SPECIES(Iele).temp;   
   elseif G{ii}.SPCGENERAL.beta_type=='eq'
     % look for the value applied for beta in the out file produced by GKW
     % would be better to have it as proper GKW output..
     str_to_find = '*****\ Using\ beta\ =\ '; 
     flpth_out = gkwpath('out',proj);
     [dum,dum_str]=unix(['grep ' str_to_find ' ' flpth_out flist{ii}]);
     dum=regexp(dum_str,['(' str_to_find ')(?<token>[^\n]*)'],'names');
     if isempty(dum)||(str2num(dum(end).token)<1 | str2num(dum(end).token)>1e6)
      is_ok(ii)=0;
      ok_msg{ii}='Problem when retrieving beta from the output file';
      continue
     else
       out{iN}.species_global.beta = str2num(dum(end).token).*Brat.^2.*G{ii}.SPECIES(Iele).dens.*G{ii}.SPECIES(Iele).temp;             
     end
   else 
      is_ok(ii)=0;
      ok_msg{ii}='Unknown beta_type option';
      continue
   end
 else
   out{iN}.species_global.beta = 0;   
 end

 %%% CF effects %%%
 
 if (G{ii}.ROTATION.cf_trap==1)|(G{ii}.ROTATION.cf_drift==1)
  if ((G{ii}.ROTATION.cf_trap+G{ii}.ROTATION.cf_drift+G{ii}.ROTATION.cf_upphi+G{ii}.ROTATION.cf_upsrc)==4)
    out{iN}.code.include_centrifugal_effects = 1;
    flpth_cfdens=gkwpath('cfdens',proj);
    cfdens=load([flpth_cfdens flist{ii}]);
    Imid = find(Ggeom{ii}.s_grid>-0.5 & Ggeom{ii}.s_grid<0.5 & Ggeom{ii}.r.*Rrat>1);    
    nefac_th0=interpos(Ggeom{ii}.z(Imid).*Rrat,cfdens(Imid,Iele+1+nsp),Z0,0.); 
    out{iN}.species_global.collisionality = out{iN}.species_global.collisionality.*nefac_th0;
    nrat = nrat.*nefac_th0;
    for jj=1:nsp % loop over species
      nsfac_th0 = interpos(Ggeom{ii}.z(Imid).*Rrat,cfdens(Imid,jj+1+nsp),Z0,0.);   % interpolation at Z=Z0 (theta=0)
      out{iN}.species{jj}.density = out{iN}.species{jj}.density.*nsfac_th0./nefac_th0; %correction needed if theta<>0 at s=0 + CF effects
      rlns_th0 = interpos(Ggeom{ii}.z(Imid).*Rrat,cfdens(Imid,jj+1),Z0,0.);
      out{iN}.species{jj}.density_log_gradient = rlns_th0./Rrat;
    end
  else
    is_ok(ii)=0;
    ok_msg{ii}='Inconsistent CF switches: all or none need to be ON for the GKDB';
    continue
  end
 else
  out{iN}.code.include_centrifugal_effects = 0;  
 end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Outputs

 for kk=1:nb_eiv   % for initial value runs, nb_eiv=1

 %%% growth rate and mode frequency %%%
 out{iN}.wavevectors{jN}.eigenvalues{kk}.growth_rate = gamma(ii).*vthrat./Rrat; 
 out{iN}.wavevectors{jN}.eigenvalues{kk}.frequency = sj_gkw.*freq(ii).*vthrat./Rrat;
 
 dum=load([gkwpath('time',proj) flist{ii}]);
 t=dum(:,1);
 g=dum(:,2);
 Delta_t=3./gamma(ii);
 I=iround(t,t(end)-Delta_t);
 out{iN}.wavevectors{jN}.eigenvalues{kk}.growth_rate_tolerance=sqrt(trapz(t(I:end),(g(I:end)-gamma(ii)).^2)./Delta_t)./gamma(ii);
 if out{iN}.wavevectors{jN}.eigenvalues{kk}.growth_rate_tolerance==0
  out{iN}.wavevectors{jN}.eigenvalues{kk}.growth_rate_tolerance=1e-6;
 end
 %%% eigenfunctions %%%
 n_turn = 2*G{ii}.GRIDSIZE.nperiod-1;
 ns_per_turn = G{ii}.GRIDSIZE.n_s_grid ./ n_turn;
 if rem(ns_per_turn,1)~=0
    is_ok(ii)=0;
    ok_msg{ii}='Number of s points per poloidal turns is not an integer';
    continue
 end
 R_dum = Ggeom{ii}.r.*Rrat;
 Z_dum = Ggeom{ii}.z.*Rrat;
 a_dum = sqrt( (R_dum - R0).^2 + (Z_dum - Z0).^2); 
 th = atan2(-(Z_dum-Z0)./a_dum,(R_dum-R0)./a_dum);
 turn_number = floor([0:G{ii}.GRIDSIZE.n_s_grid-1]./ns_per_turn);
 th_c = th + pi*(n_turn-1) - 2*pi *transpose(turn_number);
 [th_final Ith]= sort(th_c);

 % Use the kykxs diagnostic if available, parallel.dat otherwise
 test_phi=isfield(G{ii}.DIAGNOSTIC,'kykxs_phi')&&G{ii}.DIAGNOSTIC.kykxs_phi==1;
 test_apar= ~G{ii}.CONTROL.nlapar | isfield(G{ii}.DIAGNOSTIC,'kykxs_apar')&&G{ii}.DIAGNOSTIC.kykxs_apar==1;
 test_bpar= ~G{ii}.CONTROL.nlbpar | isfield(G{ii}.DIAGNOSTIC,'kykxs_bpar')&&G{ii}.DIAGNOSTIC.kykxs_bpar==1;
 fields_prefix={'phi_','apar_','bpar_'};
 fields_normfac=[rhorat.*Trat./(Rrat.*qrat) rhorat.^2.*Brat./Rrat rhorat.*Brat./Rrat];
 if  test_phi & test_apar & test_bpar % use kykxs diagnostics
   flpth_fields=gkwpath('kykxs_fields',proj);
   gkw_fields_names={'Phi','Apa','Bpa'};
   for ff=1:length(fields_prefix) 
     eval([fields_prefix{ff} 'gkw=0.*phi{' num2str(ii) '}(:,1);']);
     eval([fields_prefix{ff} 'gkdb=0.*phi{' num2str(ii) '}(:,1);']);
     [dum fl_list] = unix(['ls -x ' flpth_fields flist{ii} '/' gkw_fields_names{ff} '_kykxs*_real']);
     fl_num=regexp(fl_list,['(' gkw_fields_names{ff} '_kykxs\d+)_real'],'tokens');
     if ~isempty(fl_num) 
       fid=fopen([flpth_fields flist{ii} '/' fl_num{end}{1} '_real']);
       frewind(fid);
       dum_re=fread(fid,'double');
       fclose(fid);
       fid=fopen([flpth_fields flist{ii} '/' fl_num{end}{1} '_imag']);
       frewind(fid);
       dum_im=fread(fid,'double');
       fclose(fid);
       eval([fields_prefix{ff} 'gkw=dum_re+i.*dum_im;']);
       eval([fields_prefix{ff} 'gkdb=(dum_re(Ith)+i.*dum_im(Ith)).*fields_normfac(' num2str(ff) ');']);
     end
   end
 else% use parallel.dat
   phi_gkw   = phi{ii}(:,1)+i.*phi{ii}(:,2);
   apar_gkw  = apar{ii}(:,1)+i.*apar{ii}(:,2);
   bpar_gkw  = bpar{ii}(:,1)+i.*bpar{ii}(:,2);
   phi_gkdb  = (phi{ii}(Ith,1) + i.*phi{ii}(Ith,2)).*fields_normfac(1);
   apar_gkdb = (apar{ii}(Ith,1) + i.*apar{ii}(Ith,2)).*fields_normfac(2);
   bpar_gkdb = (bpar{ii}(Ith,1) + i.*bpar{ii}(Ith,2)).*fields_normfac(3);
   % Unrotate if necessary
   if G{ii}.DIAGNOSTIC.lrotate_parallel==1
     str_to_find = 'parallel.dat:\ normalized\ \(rotated\ in\ complex\ plane\)\ by'; 
     flpth_out = gkwpath('out',proj);
     [dum,dum_str]=unix(['grep -A 1 ' str_to_find ' ' flpth_out flist{ii}]);
     dum_str=regexprep(dum_str,'[ ]*','');
     dum=regexp(dum_str,'\((?<re>[-\d]+.*),(?<im>.*)\)','names');
     if isempty(dum)
       is_ok(ii)=0;
       ok_msg{ii}='Problem when retrieving the factor used to rotate the fields in parallel.dat';
       continue
     else
       rotate=str2num(dum.re)+i*str2num(dum.im);      
     end
     phi_gkw=phi_gkw.*rotate;
     apar_gkw=apar_gkw.*rotate;
     bpar_gkw=bpar_gkw.*rotate;
     phi_gkdb=phi_gkdb.*rotate;
     apar_gkdb=apar_gkdb.*rotate;
     bpar_gkdb=bpar_gkdb.*rotate;
   end
 end

 amp_gkw = sqrt(trapz(Ggeom{ii}.s_grid,abs(phi_gkw).^2+abs(apar_gkw).^2+abs(bpar_gkw).^2)./(Ggeom{ii}.s_grid(end)-Ggeom{ii}.s_grid(1)));
 amp_gkdb = sqrt(trapz(th_final,abs(phi_gkdb).^2+abs(apar_gkdb).^2+abs(bpar_gkdb).^2))./(2*pi);
 amp_rat = amp_gkw./amp_gkdb;

 phi_gkdb = phi_gkdb.*amp_rat;
 apar_gkdb = apar_gkdb.*amp_rat;
 bpar_gkdb = bpar_gkdb.*amp_rat;
 
 % Rotate again, following the GKDB convention
 dum_re=interpos(th_final,real(phi_gkdb),[0 0],0.);
 dum_im=interpos(th_final,imag(phi_gkdb),[0 0],0.);
 rotate_new=dum_re(1)+i.*dum_im(1);
 if abs(rotate_new)>1e-14
   rotate_new=rotate_new./abs(rotate_new);
 else
   rotate_new=1;
 end
 phi_gkdb=phi_gkdb./rotate_new;
 apar_gkdb=apar_gkdb./rotate_new;
 bpar_gkdb=bpar_gkdb./rotate_new;

 out{iN}.wavevectors{jN}.eigenvalues{kk}.eigenvector.poloidal_angle = th_final;
 out{iN}.wavevectors{jN}.eigenvalues{kk}.eigenvector.r_phi_potential_perturbed = real(phi_gkdb);
 out{iN}.wavevectors{jN}.eigenvalues{kk}.eigenvector.i_phi_potential_perturbed = imag(phi_gkdb);
 out{iN}.wavevectors{jN}.eigenvalues{kk}.eigenvector.r_a_parallel_perturbed = real(apar_gkdb);
 out{iN}.wavevectors{jN}.eigenvalues{kk}.eigenvector.i_a_parallel_perturbed = imag(apar_gkdb);
 out{iN}.wavevectors{jN}.eigenvalues{kk}.eigenvector.r_b_field_parallel_perturbed = real(bpar_gkdb);
 out{iN}.wavevectors{jN}.eigenvalues{kk}.eigenvector.i_b_field_parallel_perturbed = imag(bpar_gkdb);


 %%% moments %%%
 flpth_moments={gkwpath('kykxs_moments',proj),gkwpath('kykxs_j0_moments',proj)};
 gkw_moments_names={'dens','vpar','Tpar','Tperp','dens_ga','vpar_ga','Tpar_ga','Tperp_ga'};
 for jj=1:nsp
  if jj<10
   str=['0' num2str(jj)];
  else
   str=num2str(jj);
  end
  vthsOvthref=sqrt(Trat./mrat).*sqrt(G{ii}.SPECIES(jj).temp./G{ii}.SPECIES(jj).mass);
  TsOTref=G{ii}.SPECIES(jj).temp.*Trat;
  moments_normfac=[1 vthsOvthref TsOTref TsOTref 1 vthsOvthref TsOTref TsOTref].*amp_rat.*rhorat./Rrat.*nrat.*G{ii}.SPECIES(jj).dens;
  for mm=1:length(gkw_moments_names)
   [dum fl_list] = unix(['ls -x ' flpth_moments{1+floor((mm-1)/4)} flist{ii} '/' gkw_moments_names{mm} '_kykxs' str '*_real']);
   fl_num=regexp(fl_list,['(' gkw_moments_names{mm} '_kykxs\d+_\d+)_real'],'tokens');
   if ~isempty(fl_num) 
    fid=fopen([flpth_moments{1+floor((mm-1)/4)} flist{ii} '/' fl_num{end}{1} '_real']);
    frewind(fid);
    dum_re=fread(fid,'double');
    fclose(fid);
    fid=fopen([flpth_moments{1+floor((mm-1)/4)} flist{ii} '/' fl_num{end}{1} '_imag']);
    frewind(fid);
    dum_im=fread(fid,'double');
    fclose(fid);
    out{iN}.moments_rotating.(['r_' gkdb_moments_names{mm}])(:,jj,jN,kk) = dum_re.*moments_normfac(mm);
    out{iN}.moments_rotating.(['i_' gkdb_moments_names{mm}])(:,jj,jN,kk) = dum_im.*moments_normfac(mm);
   end
  end
 end 


 %%% fluxes %%%
 pflux_norm_base = amp_rat.^2.*rhorat.^2./Rrat.^2.*vthrat.*nrat;
 vflux_norm_base = -amp_rat.^2.*rhorat.^2./Rrat.^2.*vthrat.*nrat.*Rrat;
 eflux_norm_base = amp_rat.^2.*rhorat.^2./Rrat.^2.*vthrat.*nrat;

 for jj=1:nsp % loop over species

   pflux_norm_sp = G{ii}.SPECIES(jj).dens;
   vflux_norm_sp = G{ii}.SPECIES(jj).dens.*sqrt(out{iN}.species{jj}.mass.*out{iN}.species{jj}.temperature);
   eflux_norm_sp = G{ii}.SPECIES(jj).dens.*out{iN}.species{jj}.temperature;

   dims_fl = size(pflux.es); 
   dum=reshape(pflux.es,[],dims_fl(end));
   out{iN}.particle_fluxes.phi_potential(jj,jN,kk) = dum(ii,jj).*pflux_norm_base.*pflux_norm_sp;
   dum=reshape(pflux.em,[],dims_fl(end));
   out{iN}.particle_fluxes.a_parallel(jj,jN,kk) = dum(ii,jj).*pflux_norm_base.*pflux_norm_sp;
   dum=reshape(pflux.bpar,[],dims_fl(end));
   out{iN}.particle_fluxes.b_field_parallel(jj,jN,kk) = dum(ii,jj).*pflux_norm_base.*pflux_norm_sp;

   dims_fl = size(vflux.eslab); 
   dum=reshape(vflux.eslab,[],dims_fl(end));
   out{iN}.momentum_fluxes_lab.phi_potential(jj,jN,kk) = dum(ii,jj).*vflux_norm_base.*vflux_norm_sp;
   dum=reshape(vflux.emlab,[],dims_fl(end));
   out{iN}.momentum_fluxes_lab.a_parallel(jj,jN,kk) = dum(ii,jj).*vflux_norm_base.*vflux_norm_sp;
   dum=reshape(vflux.bparlab,[],dims_fl(end));
   out{iN}.momentum_fluxes_lab.b_field_parallel(jj,jN,kk) = dum(ii,jj).*vflux_norm_base.*vflux_norm_sp;

   dims_fl = size(vflux.es); 
   dum=reshape(vflux.es,[],dims_fl(end));
   out{iN}.momentum_fluxes_rotating.phi_potential(jj,jN,kk) = dum(ii,jj).*vflux_norm_base.*vflux_norm_sp;
   dum=reshape(vflux.em,[],dims_fl(end));
   out{iN}.momentum_fluxes_rotating.a_parallel(jj,jN,kk) = dum(ii,jj).*vflux_norm_base.*vflux_norm_sp;
   dum=reshape(vflux.bpar,[],dims_fl(end));
   out{iN}.momentum_fluxes_rotating.b_field_parallel(jj,jN,kk) = dum(ii,jj).*vflux_norm_base.*vflux_norm_sp;

   dims_fl = size(eflux.eslab); 
   dum=reshape(eflux.eslab,[],dims_fl(end));
   out{iN}.heat_fluxes_lab.phi_potential(jj,jN,kk) = dum(ii,jj).*eflux_norm_base.*eflux_norm_sp;
   dum=reshape(eflux.emlab,[],dims_fl(end));
   out{iN}.heat_fluxes_lab.a_parallel(jj,jN,kk) = dum(ii,jj).*eflux_norm_base.*eflux_norm_sp;
   dum=reshape(eflux.bparlab,[],dims_fl(end));
   out{iN}.heat_fluxes_lab.b_field_parallel(jj,jN,kk) = dum(ii,jj).*eflux_norm_base.*eflux_norm_sp;

   dims_fl = size(eflux.es); 
   dum=reshape(eflux.es,[],dims_fl(end));
   out{iN}.heat_fluxes_rotating.phi_potential(jj,jN,kk) = dum(ii,jj).*eflux_norm_base.*eflux_norm_sp;
   dum=reshape(eflux.em,[],dims_fl(end));
   out{iN}.heat_fluxes_rotating.a_parallel(jj,jN,kk) = dum(ii,jj).*eflux_norm_base.*eflux_norm_sp;
   dum=reshape(eflux.bpar,[],dims_fl(end));
   out{iN}.heat_fluxes_rotating.b_field_parallel(jj,jN,kk) = dum(ii,jj).*eflux_norm_base.*eflux_norm_sp;

  end
 end
 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Meta-data

 
 out{iN}.point.creator='Camenen';
 out{iN}.point.comment=['scan=' flnm ', proj=' proj '\n' comments];

 out{iN}.code.name='GKW';
 out{iN}.code.parameters='';
 if isempty(G{ii}.version)
  is_ok(ii)=0;
  ok_msg{ii}='Code version not known';
  continue
 else
  out{iN}.code.version=G{ii}.version;
 end

 else 

  is_ok(ii)=0;
  ok_msg{ii}='Not converged, unstable or Nperiod too small';
  continue

 end 

end % loop on wavevectors

 if ~any(is_ok(II))
  is_ok_out(iN)=0;
 end

end % loop on output files 


%%%%%%%%%%%%%%%%%%
% Checks
%%%%%%%%%%%%%%%%%%


% clean the output structure
for iN=1:Nout % loop over output files
 II=find(ii_out==iN);
 Iwok=find(is_ok(II));
 if isempty(Iwok)
   out{iN}=[];
 else
   out{iN}.particle_fluxes.phi_potential=out{iN}.particle_fluxes.phi_potential(:,Iwok,:);
   out{iN}.particle_fluxes.a_parallel=out{iN}.particle_fluxes.a_parallel(:,Iwok,:);
   out{iN}.particle_fluxes.b_field_parallel=out{iN}.particle_fluxes.b_field_parallel(:,Iwok,:);

   out{iN}.momentum_fluxes_lab.phi_potential=out{iN}.momentum_fluxes_lab.phi_potential(:,Iwok,:);
   out{iN}.momentum_fluxes_lab.a_parallel=out{iN}.momentum_fluxes_lab.a_parallel(:,Iwok,:);
   out{iN}.momentum_fluxes_lab.b_field_parallel=out{iN}.momentum_fluxes_lab.b_field_parallel(:,Iwok,:);

   out{iN}.momentum_fluxes_rotating.phi_potential=out{iN}.momentum_fluxes_rotating.phi_potential(:,Iwok,:);
   out{iN}.momentum_fluxes_rotating.a_parallel=out{iN}.momentum_fluxes_rotating.a_parallel(:,Iwok,:);
   out{iN}.momentum_fluxes_rotating.b_field_parallel=out{iN}.momentum_fluxes_rotating.b_field_parallel(:,Iwok,:);

   out{iN}.heat_fluxes_lab.phi_potential=out{iN}.heat_fluxes_lab.phi_potential(:,Iwok,:);
   out{iN}.heat_fluxes_lab.a_parallel=out{iN}.heat_fluxes_lab.a_parallel(:,Iwok,:);
   out{iN}.heat_fluxes_lab.b_field_parallel=out{iN}.heat_fluxes_lab.b_field_parallel(:,Iwok,:);

   out{iN}.heat_fluxes_rotating.phi_potential=out{iN}.heat_fluxes_rotating.phi_potential(:,Iwok,:);
   out{iN}.heat_fluxes_rotating.a_parallel=out{iN}.heat_fluxes_rotating.a_parallel(:,Iwok,:);
   out{iN}.heat_fluxes_rotating.b_field_parallel=out{iN}.heat_fluxes_rotating.b_field_parallel(:,Iwok,:);

   for mm=1:length(gkdb_moments_names) 
    out{iN}.moments_rotating.(['r_' gkdb_moments_names{mm}]) = out{iN}.moments_rotating.(['r_' gkdb_moments_names{mm}])(:,:,Iwok,:);
    out{iN}.moments_rotating.(['i_' gkdb_moments_names{mm}]) = out{iN}.moments_rotating.(['i_' gkdb_moments_names{mm}])(:,:,Iwok,:);
   end

   dum={out{iN}.wavevectors{Iwok}};
   out{iN}=rmfield(out{iN},'wavevectors');
   out{iN}.wavevectors=dum;
 end
end

% order fields
top_fields_order={'point','code','flux_surface','species','species_global','wavevectors','particle_fluxes','momentum_fluxes_lab','momentum_fluxes_rotating','heat_fluxes_lab','heat_fluxes_rotating','moments_rotating'};
for iN=1:Nout % loop over output files
  if ~isempty(out{iN})
    out{iN}=orderfields(out{iN},top_fields_order);
  end
end

%%%%%%%%%%%%%%%%%%
% write JSON files
%%%%%%%%%%%%%%%%%%
for iN=1:Nout % loop over output files
 if is_ok_out(iN)
%  savejson('',out{iN},[flpth_json flnm '_' num2str(iN) '.json']);
   writejson(out{iN},[flpth_json flnm '_' num2str(iN) '.json']);
 end
end