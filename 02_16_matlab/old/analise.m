clear
tic
javaaddpath('C:\Users\trash\Documents\MATLAB\infodynamics.jar');

num_exames = 870;
num_rois = 200;
colNames = {};

resultados = []

for exame = 0:0
    exame
    data = readmatrix(sprintf('%d.csv', exame));
    resultado = [];
    for i = 1:size(data, 2)
        for j = i:size(data, 2)
            if i == j
                resultado = [resultado, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]];
            else
                vetor_i = data(:, i)';
                vetor_j = data(:, j)';
                matriz = [vetor_i;vetor_j];
                if ~any(vetor_i(~isnan(vetor_i))) || ~any(vetor_j(~isnan(vetor_j)))
                    resultado = [resultado, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]];
                else
                    atoms = PhiIDFull(matriz);
                    resultado = [resultado, [atoms.rtr, atoms.rtx, atoms.rty, atoms.rts, atoms.xtr, atoms.xtx, atoms.xty, atoms.xts, atoms.ytr, atoms.ytx, atoms.yty, atoms.yts, atoms.str, atoms.stx, atoms.sty, atoms.sts]];
                end
            end
        end
    end
    resultados(exame+1, :) = resultado;
    resultado = [];
end

sufixos = {'rtr', 'rtx', 'rty', 'rts', 'xtr', 'xtx', 'xty', 'xts', 'ytr', 'ytx', 'yty', 'yts', 'str', 'stx', 'sty', 'sts'};
colNames = cell(1, num_rois*(num_rois+1)/2*16);
idx = 1;
for i = 1:num_rois
 for j = i:num_rois
     prefixo = sprintf('%d_%d-', i, j);
     for k = 1:16
         colNames{idx} = [prefixo sufixos{k}];
         idx = idx + 1;
     end
 end
end

toc

resultados = array2table(resultados, 'VariableNames', colNames)
writetable(resultados, 'resultados.csv');
