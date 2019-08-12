function params = preprocNonLinearOut_GetMetaParams(argNum)
% Usage: params = preprocNonLinearOut_GetMetaParams(argNum)
% 
% Get preset arguments for preprocNonLinearOut
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

params.class = 'preprocNonLinearOut';
switch argNum
    case 1
        % Original params recommended by SN
        params.gainControl = []; % Broken: gain control for each channel based on luminance /color.
        params.gainControlOut = []; % Broken: gain control for each channel based on luminance /color.
        params.nonLinOutExp = 'log'; % Output nonlinearity
        params.nonLinOutParam = 1.0000e-05; % delta to add to channel values to prevent log(0) = -inf
    case 2
        % Original params recommended by SN
        params.gainControl = []; % Broken: gain control for each channel based on luminance /color.
        params.gainControlOut = []; % Broken: gain control for each channel based on luminance /color.
        params.nonLinOutExp = .5; % Output nonlinearity
        %params.nonLinOutParam = 1.0000e-05; % delta to add to channel values to prevent log(0) = -inf
        
end
