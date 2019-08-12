function varargout = preprocDownsample(S, params)
% Usage: [Spp, params] = preprocDownsample(S, params)
% 
% Downsamples stimuli to (fMRI or other) sampling rate; e.g., takes frames
% of a movie presented at 15 Hz and downsamples them to an fMRI sampling
% rate (TR) of .5 Hz (2 seconds / measurement). 
% 
% Inputs: 
%   S = preprocessed stimulus matrix, [time x channels]
%   params = parameter struct, with fields:
%       .dsType = string specifying type of downsampling: 'box' [default],
%           'gauss', 'max', or 'none'
%       .gaussParams = 2-element array specifying [,] (??). Only necessary
%           if params.dsType = 'gauss'
%       .imHz = frame rate of stimulus in Hz 
%       .sampleSec = length in seconds of 1 sample of data (for fMRI, this
%           is the TR or repetition time)
%       .frameshifts = amount to shift frames; empty implies no shift
%       .gaussParams = standard deviation and temporal offset for Gaussian
%           downsampling window
% Output:
%   Spp
%
% Copyright (c) 2017, Regents of the University of California
% All rights reserved.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% Default parameters
dParams.dsType = 'box';
dParams.imHz = 15;
dParams.sampleSec = 2;
dParams.frameshifts = []; % empty = no shift
dParams.gaussParams = []; %[1,2]; % sigma, offset
% Fill in default params
if ~exist('params','var')
    params = struct;
end
params = defaultOpt(params,dParams);
% Return params if no inputs
if ~nargin
    varargout{1} = params;
    return
end

% Set fr_per_sample to sixteen instead of calculating it as that will
% result in 16.002, and we need a integer to work with.
%fr_per_sample = params.sampleSec*params.imHz;
fr_per_sample = 16; 

% downsample the preprocessed stimuli
switch params.dsType
    case 'box'
        if isfield(params,'frameshifts') && ~isempty(params.frameshifts)
          fprintf('shifting %d frames...\n', params.frameshifts);
          S=circshift(S,[params.frameshifts 0]);
        end
        tframes = floor(size(S,1)/fr_per_sample)*fr_per_sample;
        S = S(1:tframes,:);
        S = reshape(S, fr_per_sample, [], size(S,2));
        S = reshape(mean(S,1), [], size(S,3));
    case 'none'
        0; % do nothing
    case 'max'
        tframes = floor(size(S,1)/fr_per_sample)*fr_per_sample;
        S = S(1:tframes,:);
        S = reshape(S, fr_per_sample, [], size(S,2));
        S = reshape(max(S,[],1), [], size(S,3));
    case 'gauss'
        ksigma = params.gaussParams(1);
        if ksigma~=0
            ki = -ksigma*2.5:1/fr_per_sample:ksigma*2.5;
            k = exp(-ki.^2/(2*ksigma^2));
            S = conv2(S, k'/sum(k), 'same');
        end
        sonset = 7;
        if length(params.gaussParams)>=2
            sonset = params.gaussParams(2);
        end
        S = S(sonset:fr_per_sample:end,:);
end

% Output
varargout{1} = S;
if nargout>1
    varargout{2} = params;
end
