function [ax1,ax2,lineLossRec1,lineLossRec2,lineLoss]=initializePlots4()   %定义用于训练过程显示的函数

set(0,'defaultfigurecolor','w')

% Initialize training progress plot.
% fig1 = figure;

ax1 = subplot(2,3,1:3);

% Plot the three losses on the same axes.
hold on
lineLossRec1 = animatedline('Color','g');
lineLossRec2 = animatedline('Color','r');
lineLoss = animatedline('Color','b');

% Customize appearance of the graph.
legend('Rec1 loss','Rec2 loss','Total loss','Location','Southwest');
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

% Initialize image plot.
ax2 = subplot(2,3,4:6);
axis off

end