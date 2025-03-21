% 기본 폰트를 Times New Roman으로 설정
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');

% CSV 데이터 읽기: height50_data.csv에서 한 행 추출 (예: 35번째 행)
dataCell = cell(1, 2); 
data_ls = [50, 70];
result_50 = readmatrix('data/height50_data.csv');
for i = 1:2
    filename = sprintf('height%d_data.csv', data_ls(i));
    dataCell{i} = readmatrix(filename);
end
test_GUs = dataCell{1}(35,1:12);
gnd = reshape(test_GUs, [3, 4]);  % 3×4 행렬 (각 열: [x; y; z])
uav_pos1 = dataCell{1}(35,13:15); 
uav_pos2 = dataCell{2}(35,13:15);

% 2×3 서브플롯 생성
rows = 2;
cols = 3;
ax = gobjects(rows, cols);
for r = 1:rows
    for c = 1:cols
        index = (r-1)*cols + c;
        ax(r, c) = subplot(rows, cols, index, 'Projection', 'perspective');
    end
end

% 각 서브플롯에 대해 plotScene 호출
% 예시: 첫 행은 height 50, 두번째 행은 height 70 (여기서는 같은 데이터 사용)
plotScene(ax(1,1), 'mesh_', 0:4, gnd, uav_pos1, [-100,100], [-100,100], [0,100], [30,45]);
title(ax(1,1), 'height=50, az=30, el=45');

plotScene(ax(1,2), 'mesh_', 0:4, gnd, uav_pos1, [-100,100], [-100,100], [0,100], [280,10]);
title(ax(1,2), 'height=50, az=280, el=10');

plotScene(ax(1,3), 'mesh_', 0:4, gnd, uav_pos1, [-100,100], [-100,100], [0,100], [0,90]);
title(ax(1,3), 'height=50, az=0, el=90');

plotScene(ax(2,1), 'mesh_', 0:4, gnd, uav_pos2, [-100,100], [-100,100], [0,100], [30,45]);
title(ax(2,1), 'height=70, az=30, el=45');

plotScene(ax(2,2), 'mesh_', 0:4, gnd, uav_pos2, [-100,100], [-100,100], [0,100], [280,10]);
title(ax(2,2), 'height=70, az=280, el=10');

plotScene(ax(2,3), 'mesh_', 0:4, gnd, uav_pos2, [-100,100], [-100,100], [0,100], [0,90]);
title(ax(2,3), 'height=70, az=0, el=90');

% 전체 figure 공통 제목 추가
sgtitle('3D Mesh Scenes with UAV-Ground Connections');
