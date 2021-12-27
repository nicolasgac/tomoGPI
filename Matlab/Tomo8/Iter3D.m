%CLASS_INTERFACE Example MATLAB class wrapper to an underlying C++ class
%CLASS_INTERFACE Example MATLAB class wrapper to an underlying C++ class


classdef Iter3D < handle
    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ class instance
    end
    properties
        
        methode;
        fc;
        storage;
        workdirectory;
        precision;
        num_iter;
        calcul_eqm;
        save_volume;
        save_file;
        J;
        J_reg;
        J_MC;
        alpha;
        eam_g;
        eam_relatif_g;
        eqm_g;
        eqm_relatif_g;
        no_display_fig;
    end
    methods
        %% Constructor - Create a new C++ class instance
        function this = Iter3D(varargin)
            display(varargin{1});
            [this.objectHandle] = TomoGPI_iter_mex('new', varargin{:});
            %this.storage=storage;
            %this.precision=precision;
            this.workdirectory=varargin{1};
        end
        
        %% Destructor - Destroy the C++ class instance
        function delete(this)
            TomoGPI_iter_mex('delete', this.objectHandle,this.workdirectory);
        end
        
        function varargout = getVersion(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getVersion',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        function varargout = CreateVolumeInit(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('CreateVolumeInit',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        function varargout = CreateVolumeReal(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('CreateVolumeReal',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        function varargout = getSinoReal(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getSinoReal',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        function varargout = getOutputDirectory(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getOutputDirectory',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        
        function varargout = getDelta_un(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getDelta_un',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        
        function varargout = getLambda(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getLambda',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        function varargout = setLambda(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('setLambda',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        
        function varargout = getPositivity(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getPositivity',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        function varargout = setPositivity(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('setPositivity',this.objectHandle,this.workdirectory, varargin{:});
        end
        function varargout = getNoiseValue(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getNoiseValue',this.objectHandle,this.workdirectory, varargin{:});
        end
        function varargout = getGradientIterationNb(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getGradientIterationNb',this.objectHandle,this.workdirectory, varargin{:});
        end
        function varargout = getGlobalIterationNb(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getGlobalIterationNb',this.objectHandle,this.workdirectory, varargin{:});
        end
        function varargout = getOptimalStepIterationNb(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getOptimalStepIterationNb',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        %% doProjection - an example class method call
        function varargout = doProjection(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('doProjection',this.objectHandle,this.workdirectory, varargin{:});
        end
        
       
        
        %% Laplacian
        function varargout = ApplyLaplacianRegularization_to_dJ(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('ApplyLaplacianRegularization_to_dJ',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        
        function varargout = doLaplacian(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('doLaplacian',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        
        %% doProjection - an example class method call
        function varargout = doGradient(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('doGradient',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        %% doProjection - an example class method call
        function varargout = addNoise(this, varargin)
            rsb_dB_in=getNoiseValue(this);
            varargout{2}=rsb_dB_in;
            display(rsb_dB_in);
            if (rsb_dB_in ~= 0)
                [varargout{3},varargout{1},rsb_dB_out]=iter3D_add_noise_r_TOMO8(varargin{1},rsb_dB_in);
                display(['noise ' rsb_dB_out ' db added']);
            else
                display(['no noise added']);
            end
            
            clear b;
        end
        
        %% Filter ramp
        function varargout = doFilter_ramp(this, varargin)
            delta_un=getDelta_un(this);
            [varargout{1}]=ramp_filter_TOMO8(varargin{1},16,varargin{2}, delta_un);
        end
        
        %% doBackprojection - an example class method call
        function varargout = doFDK(this, varargin)
            sino_filtred=doFilter_ramp(this,varargin{1},varargin{2});
            [varargout{1:nargout}] = TomoGPI_iter_mex('doBackprojection_FDK',this.objectHandle,this.workdirectory, sino_filtred);
        end
        %% FDK
        function varargout = doBackprojection_FDK(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('doBackprojection_FDK',this.objectHandle,this.workdirectory, varargin{:});
        end
        %% doBackprojection - an example class method call
        function varargout = doBackprojection(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('doBackprojection',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        
        %% attributes
        function varargout = getXVolumePixelNb(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getXVolumePixelNb',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        
        function varargout = getYVolumePixelNb(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getYVolumePixelNb',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        
        function varargout = getZVolumePixelNb(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getZVolumePixelNb',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        
        function varargout = getUSinogramPixelNb(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getUSinogramPixelNb',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        
        function varargout = getVSinogramPixelNb(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getVSinogramPixelNb',this.objectHandle,this.workdirectory, varargin{:});
        end
        
        
        function varargout = getProjectionSinogramPixelNb(this, varargin)
            [varargout{1:nargout}] = TomoGPI_iter_mex('getProjectionSinogramPixelNb',this.objectHandle,this.workdirectory, varargin{:});
        end
    
        
    end
end
