%% Script To Generate The DataSet Used to Train & Test the ANN
clc
clear
close all
%% Reading Grid and Init
 cd 'C:\Users\XXXX\Documents\File_Name\Simulation Directory'

Grid = readtable('Inputs_Grid.csv');

Max_temp = zeros(size(Grid,1),1);
TimeDeltaOver = zeros(size(Grid,1),1);
Temp_K = convtemp(723,'C','K');
%% Data Generated 1
cd './Simulation Results/CSV_1_1890'
% Data from out_1 to out_157
count = 1;
i_A = [];
FileName = cell(1,1900);
FileRead = zeros(1,1900);
for i = 1:1900
    try
        FileName{i} = ['out_',num2str(i),'_',num2str(i+1),...
            '_',num2str(i+2),'.csv'];
        tempData=readtable(FileName{i});
        [Max_temp(count), TimeDeltaOver(count)] = ...
            calc(tempData.TIME,tempData.POINT1,Temp_K);
        [Max_temp(count+1), TimeDeltaOver(count+1)] = ...
            calc(tempData.TIME,tempData.POINT2,Temp_K);
        [Max_temp(count+2), TimeDeltaOver(count+2)] = ...
            calc(tempData.TIME,tempData.POINT3,Temp_K);
        clear tempData
        FileRead(i) = 1;
        count = count + 3;
    catch
        continue
    end
end
% %% Find Files NOT IN PATTERN
% file_ind=find(FileRead==1);
% for i=1:length(file_ind)
%     ok_files{i}=FileName{file_ind(i)};
% end
% D=dir;
% for i=1:(size(D(:),1)-3)
%     for j=1:(size(ok_files,2))
%         if strcmp(ok_files{j},D(i).name)
%             break
%         end
%         if j==(size(ok_files,2))
%         disp('File Not used:')
%         disp(D(i).name)
%         end
%     end
% end
%% Data Generated 2
cd '../CSV_1892_3644'
for i = 1892:3:3644
    FileName = ['out',num2str(i),'.csv'];
    tempData = readtable(FileName);
    [Max_temp(i-1), TimeDeltaOver(i-1)] = ...
        calc(tempData.TIME,tempData.POINT1,Temp_K);
    [Max_temp(i), TimeDeltaOver(i)] = ...
        calc(tempData.TIME,tempData.POINT2,Temp_K);
    [Max_temp(i+1), TimeDeltaOver(i+1)] = ...
        calc(tempData.TIME,tempData.POINT3,Temp_K);
    clear tempData
    count = count + 3;
end
%% Dataset Creation
cd '../../'
names = [Grid.Properties.VariableNames,{'Max Temperature'},{'Delta Time'}];
Dataset = table(Grid.PlateThickness,Grid.InitialTemperature,...
    Grid.HeatInput, Grid.ElectrodeVelocity, Grid.X, Grid.Y, Grid.Z,...
    Max_temp, TimeDeltaOver);
Dataset.Properties.VariableNames=names;

% Save
writetable(Dataset, 'dataset.csv')
%% Dataset without zeros on Max Temp
Dataset_NoZeros = Dataset(Dataset.("Max Temperature")>0,:);
writetable(Dataset_NoZeros, 'dataset_No_Zeros.csv')
%% Dataset without zeros on Max Temp
Dataset_Edited = Dataset(Dataset.("Max Temperature")>1,:);
writetable(Dataset_Edited, 'dataset_Edited.csv')
%% Function
function [maxT,deltaTime] = calc(t,Point,deltaTemp)
    maxT = max(Point);
    if isempty(find((Point>deltaTemp)==1, 1))
        deltaTime = 0;
    else
        startInd = find((Point>deltaTemp)==1, 1,'first');
        startT = t(startInd);
        stopInd = find((Point>deltaTemp)==1, 1,'last');
        stopT = t(stopInd);
        deltaTime = stopT - startT;
    end

end
