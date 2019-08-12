function varargout = preprocColorSpace(S, params)
% Usage: [Spreproc, params] = preprocColorSpace(S, params)
%
% A script for switching between color spaces for color stimuli; relies on
% color functions in Matlab's ImageProcessing toolbox. 
%
% Created by SN 2009
% Modified/commented by ML 2013.03.21
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
dParams.class = 'preprocColorSpace';
dParams.colorconv = 'rgb2lab'; % string name for matlab color function
dParams.colorchannels = 1;
dParams.gamma = 1.0;
dParams.verbose = true;
% Fill default parameters
if ~exist('params','var')
    params = struct;
end
params = defaultOpt(params,dParams);
if nargin<1 % just called to set the default set of parameters
    varargout{1} = params;
    return;
end
% assure image is in 0-255 range before gamma-correcting
if max(S(:))<1.1;
    S = S*255;
end
% Assure functions are available
if exist(params.colorconv)~=2
    error(['Selected color conversion function {%s} is not available in your matlab install!\n' ...
           'Try using params.colorconv = ''rgb2gray'' or use params = preprocColorSpace_GetMetaParams(2)\n' ...
           '(Or you may simply not have the image processing toolbox...)\n'], params.colorconv)
end
% color-space conversion
if params.verbose; 
    disp(['Converting color space ... [' params.colorconv ']']); 
end

cstim = zeros(size(S,1),size(S,2),size(S,4),length(params.colorchannels),'single');
framenum=size(S,4);

for ii=1:framenum
    tim = squeeze(S(:,:,:,ii));
    tim = gammacorrect(tim, params.gamma);
    cim = feval(params.colorconv, tim);
    cstim(:,:,ii,:) = single(cim(:,:,params.colorchannels));
    progressdot(ii,500,10000,framenum);
end
clear S;
Spreproc = cstim;
varargout{1} = Spreproc;
if nargout>1
    varargout{2} = params;
end
if params.verbose
    disp('Done.')
end


return


function im = gammacorrect(im, g)

im = double(im)/255;
if g~=1.0
    im = im.^g;
end

return


