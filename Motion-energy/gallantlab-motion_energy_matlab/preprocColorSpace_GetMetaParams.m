function params = preprocColorSpace_GetMetaParams(argNum)
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

params.class = 'preprocColorSpace';

switch argNum
    case 1
        % Convert to L*A*B colorspace, keep luminance channel
        params.class = 'preprocColorSpace';
        params.colorconv = 'rgb2lab';
        params.colorchannels = 1;
        params.gamma = 1.0;
        params.verbose = true;
    case 2
        % Convert to grayscale using rgb2gray (inferior, but present on
        % older matlab versions
        params.class = 'preprocColorSpace';
        params.colorconv = 'rgb2gray';
        params.colorchannels = 1;
        params.gamma = 1.0;
        params.verbose = true;        
    otherwise
        error('Unknown argument!')
end