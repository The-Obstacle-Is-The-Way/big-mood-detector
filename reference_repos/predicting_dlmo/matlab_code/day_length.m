function [hours, b]=day_length(Day,Latitude)
%hours=day_length(Day,Latitude)
%This calculates the number of hours (hours) and fraction of the day (b) in %daylight.
%Inputs:
% Day - day of the year, counted starting with the day of the December solstice in the first year of a Great Year.
% Latitude - latitude in degrees, North is positive, South is negative
%
%Calculations are per Herbert Glarner's formulae which do not take into account refraction, twilight, size of the sun, etc. (http://herbert.gandraxa.com/herbert/lod.asp but be careful about inconsistencies in radians/degrees).

%
%Copyright (c) 2015, Travis Wiens
%All rights reserved.
%
%Redistribution and use in source and binary forms, with or without 
%modification, are permitted provided that the following conditions are 
%met:
%
%    * Redistributions of source code must retain the above copyright 
%      notice, this list of conditions and the following disclaimer.
%    * Redistributions in binary form must reproduce the above copyright 
%      notice, this list of conditions and the following disclaimer in 
%      the documentation and/or other materials provided with the distribution
%      
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
%AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
%IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
%ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
%LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
%CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
%SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
%INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
%CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
%ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
%POSSIBILITY OF SUCH DAMAGE.

if nargin<1
  Day=0;
end
if nargin<2
  Latitude=(-(36+51/60));
end

Axis=23.439*pi/180;

j=pi/182.625;%constant (radians)
m=1-tan(Latitude*pi/180).*tan(Axis*cos(j*Day));

m(m>2)=2;%saturate value for artic
m(m<0)=0;

b=acos(1-m)/pi;%fraction of the day the sun is up


hours=b*24;%hours of sunlight