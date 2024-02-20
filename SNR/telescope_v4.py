##created by Hannah Gulick 02/09/2021
##v2 includes the a spectral radiance addition--> input parameter L_ins changed to T for
#later calculation
##v3 06/20/2021 SNR calculation includes the option to input a background source magnitude in mag/arcsec^2

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

class Telescope:

    #constants
    h = 6.63e-27*u.cm**2*u.g*u.s**-1 #cm2 g/s
    h_erg = 6.63e-27*u.erg*u.s #erg s
    k_erg = 1.38e-16*u.erg/u.K
    c = 2.99e10*u.cm/u.s #cm/s
    #tau_atm = 1. #transmissivity of atmosphere
    
    #filter coverage in angstrom (Bessell et al. 1998,  http://www.astronomy.ohio-state.edu/~martini/usefuldata.html)
    dellu = u.angstrom
    deltaLamb = {'UVA' : 850*dellu, 'UVB' : 350*dellu, 'UVC' : 1800*dellu, 'U': 600*dellu, 'B': 900*dellu, 'V': 850*dellu, 'R': 1500*dellu, 'I': 1500*dellu, 
                 'J': 2600*dellu, 'H': 2900*dellu, 'K': 4100*dellu}
    #sky background in erg/s/cm**2/angstrom/arcsec**2 https://hst-docs.stsci.edu/wfc3ihb/chapter-9-wfc3-exposure-time-calculation/9-7-sky-background
    bskyu = u.erg/u.s/(u.cm)**2/u.angstrom/(u.arcsec)**2
    Bsky_erg = {'U': 3.55E-18*bskyu, 'B': 7.57E-18*bskyu, 'V': 7.72E-18*bskyu, 'R': 7.56E-18*bskyu, 
                'I': 5.38E-18*bskyu, 'J': 2.61E-18*bskyu, 'H': 1.43E-18*bskyu, 'K': 4100*bskyu}
    #effective wavelength for filters (cm) http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    lambu = u.cm
    lamb_eff = {'UVA': 3.57e-5*lambu, 'UVB': 1.97e-5*lambu, 'UVC':  1.90e-5*lambu, 'U': 3.60e-5*lambu, 'B': 4.38e-5*lambu, 'V': 5.45e-5*lambu, 'R': 6.41e-5*lambu, 'I': 7.98e-5*lambu, 'J': 0.000122*lambu, 'H': 0.000163*lambu, 'K': 0.000219*lambu}
    
   
    #Vega zero-point flux: erg cm-2 s-1 A-1 (Bessell et al. 1998,  http://www.astronomy.ohio-state.edu/~martini/usefuldata.html)
    F0u = u.erg*(u.cm**(-2))/u.s/u.angstrom
    F0_lamb = {'U': 417e-11*F0u, 'B': 632e-11*F0u, 'V': 363.1e-11*F0u, 'R': 217.7e-11*F0u, 'I': 112.6e-11*F0u, 
               'J': 31.47e-11*F0u, 'H': 11.38e-11*F0u, 'K': 3.961e-11*F0u}
    
    ##Zero-point fluxes: Jy (from http://astroweb.case.edu/ssm/ASTR620/mags.html)
    F0lamb_AB = {'U': 1810*F0u, 'B': 4260*F0u, 'V': 3640*F0u, 'R': 3080*F0u, 'I': 2550*F0u, 
              'J': 1600*F0u, 'H': 1080*F0u, 'K': 670*F0u}
    
    #converting Vega magnitude to AB magnitude: http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    deltaMag = {'U': 0.79, 'B': -0.09, 'V': 0.02, 'R': 0.21, 'I': 0.45, 'J': 0.91, 'H': 1.39, 'K': 1.85}
    
    #put units
    def __init__(self, filt, phot_r, D, tau_ins, tau_atm, eta, T, Dark, Ron, n_pixel):
        '''This comment is a work in progress:
        
        Inputs: 
        filt: string
            a single letter for the filter name; either U, B, V, R, I, J, H, or K
        phot_r: integer or float with astropy units
            an integer value (recommended units: ") which gives the photometric radius
        D: integer or float with astropy units
            the diamter of the telescope (recommeded units: cm)
        tau_ins: integer or float
            the instrumental efficiency
        tau_atm: integer or float
            the transmissivity of the atmosphere (take to be 1 for space)
        eta: integer or float with astropy units
            the quantum efficiency of the detector (recommended units e-)
        T: temperature of the telescope with astropy units
            the instrumental radiance (recommended units: K)
        Dark: integer or float with astropy units
            the dark current (recommended units: electron/pix/s)
        Ron: integer or float with astropy units
            the RMS readout noise (recommended units: electron**0.5/s**0.5)
        n_pixel: integer or float with astropy units
            the number of pixels covered by a source (recommended units: pix)'''
        
        
        self.filt = filt
        self.phot_r = phot_r
        self.D = D
        self.tau_ins = tau_ins
        self.tau_atm = tau_atm
        self.eta = eta
        self.T = T
        self.Dark = Dark
        self.Ron = Ron
        self.n_pixel = n_pixel
        

    def spectral_radiance(self, lamb_eff = lamb_eff, deltaLamb = deltaLamb, h_erg = h_erg, c =c, k_erg = k_erg):
        '''Calculates the spectral radiance of the instrument at different wavelengths'''
        C1 = 3.7418e-5*u.erg*u.cm**2/u.s
        C2 = 1.439*u.K*u.cm
        #radiance = C1/(np.pi*lamb_eff[self.filt]**5*(np.exp(C2/(lamb_eff[self.filt]*self.T))-1))*deltaLamb[self.filt]/u.sr
        radiance = 2*h_erg*c**2/lamb_eff[self.filt]**5*(1/np.exp((h_erg*c)/(lamb_eff[self.filt]*k_erg*self.T)-1))*deltaLamb[self.filt]/u.sr
        
        #print('RADIANCE', radiance)
        
        return radiance
        
    def photon_E(self, lamb_eff=lamb_eff):
        '''Calculated photon energy'''
        h_erg = 6.63e-27*u.erg*u.s #erg s
        c = 2.99e10*u.cm/u.s #cm/s
        E_photon = h_erg*c/lamb_eff[self.filt] #erg
        return E_photon
        
    def bsky(self, lamb_eff = lamb_eff, deltaLamb = deltaLamb, Bsky_erg = Bsky_erg, h_erg = h_erg, c =c):
        '''Calculates the sky background given the wavelength, wavelength range, and sky background in the filter'''
        Bsky = Bsky_erg[self.filt]*self.tau_ins*self.eta*np.pi*(self.D/2.)**2*self.phot_r**2*(lamb_eff[self.filt]*deltaLamb[self.filt]/(h_erg*c))

        return Bsky

    def mag_to_flux(self, m1, deltaLamb = deltaLamb, F0_lamb = F0_lamb):
        '''Converts magnitude to flux for Vega magntiudes
        
        Inputs: Vega magnitude of source and filter;
        filter options include: 'U', 'B', 'V', 'R', 'I', 'J', 'H', or 'K'
        Output: Vega magnitude source flux in erg cm-2 s-1 A-1''' 
    
        F0_lamb = F0_lamb[self.filt]*deltaLamb[self.filt] #erg cm-2 s-1
        m2 = 0.0 #vega mag zero point
        F1 = F0_lamb*10**(-(m1-m2)/2.5) #erg cm-2 s-1 A-1
        return F1
    
    def mag_to_flux_AB(self, mag, F0lamb=F0lamb_AB, deltaMag=deltaMag, deltaLamb=deltaLamb, lamb_eff=lamb_eff):
        '''Converts magnitude to flux for AB magnitudes
        
        Inputs: Vega magnitude of source and filter;
        filter options include: 'U', 'B', 'V', 'R', 'I', 'J', 'H', or 'K'
        Output: AB magnitude source flux in erg cm-2 s-1 A-1'''
    
        magAB = mag+deltaMag[self.filt]
    
        F1 = F0lamb[self.filt]*10**(-0.4*magAB) *deltaLamb[self.filt]#jy
        #conversion Jy --> erg cm-2 s-1 A-1
        Flamb = 3e-5*F1/(lamb_eff[self.filt])**2 #erg cm-2 s-1 A-1 ( conversion from https://www.stsci.edu/~strolger/docs/UNITS.txt)
        return Flamb
    
    def sigma( self, m, t, b, h =h, c=c, lamb_eff=lamb_eff):
        '''Calculates sigma for all given noise or background sources'''
        
        #times = [t[i] for i in range(len(t))]
        F_sig_vega = self.mag_to_flux(m)
        F_sig_AB = self.mag_to_flux_AB(m)

        #converting galaxy magnitude in mag/arcsec^2 to mag
        b_mag = b*u.arcsec**2-2.5*np.log10(self.phot_r**2/u.arcsec**2) #mag
        #print('The background magnitude is:', b_mag)
        F_sig_vega_back = self.mag_to_flux(b_mag)
        L_ins = self.spectral_radiance()
        
        Sig = self.tau_atm*self.tau_ins*self.eta*F_sig_vega*np.pi*(self.D/2.)**2*lamb_eff[self.filt]/(h*c)*0.5 #(0.5 comes from obscuration in Frank's design)#signal electrons per second
        #Sig_back = 2.5e11*(self.tau_ins)*10**(-0.4*(b+5))*u.electron/u.s/u.pix
        Sig_back = self.tau_atm*self.tau_ins*self.eta*F_sig_vega_back*np.pi*(self.D/2.)**2*lamb_eff[self.filt]/(h*c)*u.g*u.cm**2/(u.erg*u.s**2)/(u.pix)#signal electrons per second
        #print('SIG BACK', Sig_back)
        B_ins = self.eta*L_ins*np.pi*(self.D/2.)**2*self.phot_r**2*(lamb_eff[self.filt]/(h*c))*4.8481e-6**2*u.steradian/u.arcsec**2*1e-8*u.cm/u.angstrom*u.g*u.cm**2/(u.erg*u.s**2) #electrons per second from the total instrumental background

        #sigma = [np.sqrt(Sig*t[i]+(self.bsky()*t[i]+B_ins*t[i]+self.n_pixel*(self.Dark*t[i]+self.Ron**2+Sig_back*t[i]))).to(u.electron**(1/2)) for i in range(len(t))]
        
        sigma = [np.sqrt(Sig*t[i]+t[i]*(self.bsky()+B_ins)+t[i]*self.n_pixel*(Sig_back+self.Dark)+self.n_pixel*self.Ron**2)  for i in range(len(t))]
        
        
        #print('BINS', B_ins)
        
        
        return sigma, self.bsky(), Sig, B_ins, Sig_back

    
    def s_n(self, m, t, n_f, b):
        '''Calculates the signal to noise'''

        signal = self.sigma(m,t, b)[2]
        sig = self.sigma(m,t, b)[0]
        beta = self.bsky()/self.sigma(m,t, b)[2]
        alpha1 = self.sigma(m,t, b)[3]/self.bsky()
        alpha2 = [self.n_pixel*(self.Dark*t[i]+self.Ron**2)/(self.bsky()*t[i]) for i in range(len(t))]
        
        #S_N = [np.sqrt(n_f*t[i])*np.sqrt(self.bsky()/beta)*np.sqrt(1./(1.+2.*(beta+beta*alpha1+beta*alpha2[i])))for i in range(len(t))]
        S_N = [(n_f*signal*t[i]/(np.sqrt(n_f)*sig[i])).to(u.electron**(1/2)) for i in range(len(t))]
        
        S_N_unitless = [i/(1.*(u.electron**0.5)) for i in S_N]
        t_unitless = [i/(1.*(u.s)) for i in t]
        
        #find the exposure time which is closest to 10 sigma
        arr = np.asarray(S_N_unitless)
        i = (np.abs(arr - 10)).argmin()
        #print('The profile reaches %.2f SNR at %.3f exposure time.' %( S_N_unitless[i], t_unitless[i]))
        
        return S_N, beta
    
    
    def sn_v_time_plot(self, t_low, t_high, m, n_f, b, lamb_eff=lamb_eff):
        '''Plots the SNR versus time for a user given magnitude'''
    
        times = np.linspace(t_low, t_high)
        times = [i*u.s for i in times]
        
        s_n_vega, beta_vega = self.s_n( m, times, n_f, b)
        #print('Max SNR', s_n_vega[-1])
        snvega = [i/(1.*(u.electron**0.5)) for i in s_n_vega]
        times = [i/(1.*(u.s)) for i in times]
        plt.plot(times, snvega, linestyle = 'dashed', linewidth = 5, label = self.filt+' Band')
        plt.xlabel('Time (s)', fontsize = 40)
        plt.ylabel('S/N', fontsize = 40)
        plt.xticks(size = 30)
        plt.yticks(size = 30)
        plt.title('Time versus S/N for '+str(m)+' Vega mag star', fontsize = 30)
        plt.legend(fontsize = 20)
        plt.savefig('time_sn_'+str(lamb_eff[self.filt])+'_'+str(m)+'_'+str(self.D)+'mag.png')
        plt.show()
    
        return
    
    def sn_v_mag_plot(self, mag_low, mag_high, t, n_f, b, lamb_eff=lamb_eff):
        '''Plots the SNR versus time for a user given exposure time'''
        
        times = np.array([t,0.1])
        times = [i*u.s for i in times]
        mags = np.linspace(mag_low, mag_high)

        s_n_vega, beta_vega = self.s_n( mags, times, n_f, b)
        snvega = [i/(1.*(u.electron**0.5)) for i in s_n_vega[0]]
        diff_func = lambda l: abs(l-10)
    
        SN10_vega = min(snvega, key=diff_func)
        SN10i_vega = list(snvega).index(SN10_vega)
        mag10_vega = mags[SN10i_vega]

        plt.plot(mags, snvega, linestyle = 'dashed', linewidth = 5, label = 'Vega Mag')
        plt.xlabel('Apparent Magnitude', fontsize = 40)
        plt.ylabel('S/N', fontsize = 40)
        plt.xticks(size = 30)
        plt.yticks(size = 30)
        if max(snvega) >= 10.:
            plt.axhline(10, alpha = 0.5, label = '10 sigma', color = 'magenta', linestyle = 'dashed', linewidth = 3)
            plt.axvline(mag10_vega, alpha = 0.5, label = 'Vega Mag for SN10: ' + str(round(mag10_vega, 2)), color = 'black', linestyle = 'dashed', linewidth = 3)
            plt.legend(fontsize = 30)
        plt.title('Magnitude versus S/N for ' + str(t) + ' sec Exposure', fontsize = 30)
        plt.legend()
        plt.savefig('time_sn_'+str(lamb_eff[self.filt])+'_'+str(t)+'_'+str(self.D)+'mag.png')
        plt.show()
    
        return

    def source_diagram(self, t_low, t_high, m, n_f, b, lamb_eff=lamb_eff):
        '''Plots the source signal as a function of exposure time for a fixed magnitude
            Still in progres...'''

        times = np.linspace(t_low, t_high)
        times = [i*u.s for i in times]

        sigma, bsky, Sig, B_ins, Sig_back = self.sigma(m,times, b)
         
        times = [i/(1.*(u.s)) for i in times]
        Sig = [Sig*i/u.electron/u.erg/u.s*u.cm**2*u.g for i in times]
        Sig_back = [Sig_back*i/u.electron*u.s*n_pixel for i in times]
        Ron = [self.Ron*1*u.pix**0.5/u.electron**0.5 for i in times]
        B_ins = [(B_ins/u.electron*u.s*i) for i in times]
        bsky = [(bsky*u.s/u.electron*i) for i in times]
        Dark = [(self.Dark*1*u.s*u.pix/u.electron) for i in times]
        
        plt.plot(times, Sig, linestyle = 'dashed', linewidth = 5, label = 'Source')
        plt.plot(times, Sig_back, linestyle = 'dashed', linewidth = 5, label = 'Background Galaxy')
        #plt.plot(times, np.array(Sig_back)+np.array(bsky)+np.array(B_ins), linestyle = 'dashed', linewidth = 5, alpha = 0.5, label = 'Background Galaxy + Sky + Thermal Noise')
        
        plt.xlabel('Time (s)', fontsize = 40)
        plt.ylabel('Signal or Noise (electrons)', fontsize = 40)
        plt.xticks(size = 30)
        plt.yticks(size = 30)
        plt.title('Source and Background Counts', fontsize = 30)
        plt.legend(fontsize = 20)
        plt.savefig('time_sn_'+str(lamb_eff[self.filt])+'_'+str(m)+'_'+str(self.D)+'mag.png')
        plt.show()

    def noise_diagram(self, t_low, t_high, m, n_f, b, lamb_eff=lamb_eff):
        '''Calcualte the noise profile as a function of exposure time for a fixed magnitude
            Still in progres....'''

        times = np.linspace(t_low, t_high)
        times = [i*u.s for i in times]

        sigma, bsky, Sig, B_ins, Sig_back = self.sigma(m,times, b)
        
        s_n_vega, beta_vega = self.s_n( m, times, n_f, b)
        
        times = [i/(1.*(u.s)) for i in times]
        Ron = [self.Ron*1*u.pix**0.5/u.electron**0.5 for i in times]
        B_ins = [(B_ins/u.electron*u.s*i) for i in times]
        bsky = [(bsky*u.s/u.electron*i) for i in times]
        Dark = [(self.Dark*1*u.s*u.pix/u.electron) for i in times]
        
        plt.plot(times, Ron, linestyle = 'dashed', linewidth = 5, label = 'Readout Noise')
        plt.plot(times, Dark, linestyle = 'dashed', linewidth = 5, label = 'Dark Current')
        plt.plot(times, np.array(bsky)+np.array(B_ins), linestyle = 'dashed', linewidth = 5, label = 'Sky+thermal Background')
        
        plt.xlabel('Time (s)', fontsize = 40)
        plt.ylabel('Signal or Noise (electrons)', fontsize = 40)
        plt.xticks(size = 30)
        plt.yticks(size = 30)
        plt.title('Noise Contributions', fontsize = 30)
        plt.legend(fontsize = 20)
        plt.savefig('time_sn_'+str(lamb_eff[self.filt])+'_'+str(m)+'_'+str(self.D)+'mag.png')
        plt.show()

        

