function params = preprocDownsample_GetMetaParams(Arg)
% Usage: params = preprocDownsample_GetMetaParams(Arg)
% 
% ML 2012.11.16
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

params.class = 'preprocDownsample';
switch Arg
    case 1
        % Simple box average, for TR=1, imhz=15
        params.dsType = 'box';
        params.imHz = 15;     % movie / image sequence frame rate
        params.sampleSec = 1; % TR
        params.frameshifts = []; % empty = no shift
        params.gaussParams = []; %[1,2]; % sigma,mean
    case 2
        % Simple box average, for TR=2, imhz=15
        params.dsType = 'box';
        params.imHz = 15;     % movie / image sequence frame rate
        params.sampleSec = 2; % TR
        params.frameshifts = []; % empty = no shift
        params.gaussParams = []; %[1,2]; % sigma,mean
    case 3
        % Gaussian downsampling, for TR=2, imhz=15
        params.dsType = 'gauss';
        params.imHz = 15;     % movie / image sequence frame rate
        params.sampleSec = 2; % TR
        params.frameshifts = []; % empty = no shift
        params.gaussParams = [1,2]; % mean, standard deviation
    case 4
        % Max downsampling, for TR=2, imhz=15
        params.dsType = 'max';
        params.imHz = 15;     % movie / image sequence frame rate
        params.sampleSec = 2; % TR
        params.frameshifts = []; % empty = no shift
        params.gaussParams = []; %[1,2]; % sigma,mean
    case 5
        % Simple box average, for TR=2, imhz=24
        params.dsType = 'box';
        params.imHz = 24;     % movie / image sequence frame rate
        params.sampleSec = 2; % TR
        params.frameshifts = []; % empty = no shift
        params.gaussParams = []; %[1,2]; % sigma,mean
    case 6
        % Simple box average, for TR=1, imhz=24
        params.dsType = 'box';
        params.imHz = 24;     % movie / image sequence frame rate
        params.sampleSec = 1; % TR
        params.frameshifts = []; % empty = no shift
        params.gaussParams = []; %[1,2]; % sigma,mean    
    case 7
        % Simple box average, for TR=0.7, imhz=22.86
        params.dsType = 'box';
        params.imHz = 22.86;     % movie / image sequence frame rate
        params.sampleSec = 0.7; % TR
        params.frameshifts = []; % empty = no shift
        params.gaussParams = []; %[1,2]; % sigma,mean
    otherwise
        error('Unknown parameter configuration!');
end
