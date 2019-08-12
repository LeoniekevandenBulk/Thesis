function params = preprocNormalize_GetMetaParams(argNum)
% Usage: params = preprocNormalize_GetMetaParams(argNum)
% 
% Get meta-parameters for preprocNormalize
% 
% ML 2013.03.21
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

params.class = 'preprocNormalize';
switch argNum
    case 1
        % Original arguments recmomended by SN
        params.valid_w_index = []; % specific index of channels to keep (overrides .reduceChannels)
        params.reduceChannels = []; % number of channels / pct of channels to keep
        params.normalize = 'zscore'; % normalization method
        params.crop = []; % min/max to which to crop; empty does nothing
    case 2
        params.valid_w_index = [];
        params.reduceChannels = []; %
        params.normalize = 'gaussianize';
        params.crop = [];
    case 3
        % Original arguments recmomended by SN; crops to [-3.5,3.5]
        params.valid_w_index = []; % specific index of channels to keep (overrides .reduceChannels)
        params.reduceChannels = []; % number of channels / pct of channels to keep
        params.normalize = 'zscore'; % normalization method
        params.crop = [-3.5,3.5]; % min/max to which to crop; empty does nothing
        file = load('../UnnormalizedMotionEnergyTrainingVideos/meanAndStd.mat'); % Load struct containing mean and std
        meanAndStd = file.meanAndStd; % Unpack struct
        params.means = meanAndStd.mean; % Mean of the Training Videos
        params.stds = meanAndStd.std; % Std of of the Training Videos
    case 4
        % blank for now...
    case 5
        % Original arguments recmomended by SN; crops to [-5,5]
        params.valid_w_index = []; % specific index of channels to keep (overrides .reduceChannels)
        params.reduceChannels = []; % number of channels / pct of channels to keep
        params.normalize = 'zscore'; % normalization method
        params.crop = [-5,5]; % min/max to which to crop; empty does nothing
        params.useTrnParams = true;
end