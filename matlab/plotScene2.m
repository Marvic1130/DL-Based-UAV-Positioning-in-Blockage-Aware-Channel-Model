function plotScene2(ax, stlPrefix, stlIndices, gnd, ...
    uav, xlimVal, ylimVal, zlimVal, viewAngles)
% plotScene: 지정한 축(ax)에 STL 메쉬들을 불러와 표시하고, 
%            gnd (ground points)와 uav_pos (UAV 위치)를 scatter로 표시한 후,
%            UAV와 각 gnd 포인트 사이에 선을 그려 3D 씬을 구성합니다.
%
% INPUTS:
%   ax         - 대상 축 핸들 (3D 축)
%   stlPrefix  - STL 파일 이름 앞부분 (예: 'mesh_')
%   stlIndices - STL 파일 인덱스 (예: 0:4)
%   gnd        - 3×N ground 좌표 행렬 (각 열이 하나의 점 [x;y;z])
%   uav_pos    - 1×3 UAV 좌표
%   xlimVal    - x축 범위 (예: [-100, 100])
%   ylimVal    - y축 범위 (예: [-100, 100])
%   zlimVal    - z축 범위 (예: [0, 100])
%   viewAngles - [az, el] 형태의 시야각 (방위각, 고도)
%
% 예:
%   plotScene(ax, 'mesh_', 0:4, gnd, uav_pos, [-100,100], [-100,100], [0,100], [30,45]);

    % Tab10 colormap
    tab10 = [
        31, 119, 180;
        255, 127, 14;
        44, 160, 44;
        214, 39, 40;
        148, 103, 189;
        140, 86, 75;
        227, 119, 194;
        127, 127, 127;
        188, 189, 34;
        23, 190, 207
    ] / 255;

    % 현재 축(ax) 활성화 및 hold on
    axes(ax);
    hold(ax, 'on');
    
    % STL 파일들을 순회하며 patch 그리기
    for i = stlIndices
        filename = sprintf('%s%d.stl', stlPrefix, i);
        fv = stlread(filename);
        % 각 메쉬를 patch로 표시, tab10 색상을 순서대로 사용
        patch(ax, 'Faces', fv.ConnectivityList, 'Vertices', fv.Points, ...
              'FaceColor', 'flat', 'FaceVertexCData', tab10(mod(i,10)+1, :), ...
              'EdgeColor', 'none', 'FaceAlpha', 0.8);
    end
    
    % Ground와 UAV scatter 추가
    scatter3(ax, gnd(1,:), gnd(2,:), gnd(3,:), 50, 'r', 'filled');
    scatter3(ax, uav(1), uav(2), uav(3), 50, 'g', 'filled');
    
    % UAV와 각 gnd 포인트 사이에 선 그리기
    for k = 1:size(gnd,2)
        plot3(ax, [uav(1) gnd(1,k)], [uav(2) gnd(2,k)], [uav(3) gnd(3,k)], ...
              'g-', 'LineWidth', 2);
    end
    
    % 조명 및 시각적 효과 설정
    grid(ax, 'on');
    % camlight(ax, 'right');
    % camlight(ax, 'left');
    lighting(ax, 'gouraud');
    material(ax, 'shiny');
    axis(ax, 'equal');
    
    % 축 범위 및 시야 설정
    xlim(ax, xlimVal);
    ylim(ax, ylimVal);
    zlim(ax, zlimVal);
    view(ax, viewAngles(1), viewAngles(2));
    rotate3d(ax, 'on');
    
    hold(ax, 'off');
end
