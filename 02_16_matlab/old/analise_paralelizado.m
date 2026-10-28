clear
tic
javaaddpath('C:\Users\trash\Documents\MATLAB\infodynamics.jar');
parpool('local');
jidt_path = 'C:\Users\trash\Documents\MATLAB\infodynamics.jar';
parfevalOnAll(@() javaaddpath(jidt_path), 0);

num_exames = 870;
num_rois = 200;
num_cols = (num_rois*(num_rois+1)/2) * 16;

resultados = zeros(num_exames+1, num_cols); % pré-alocação

parfor exame = 0:num_exames
    fprintf('Worker %d processando exame %d\n', getCurrentTask().ID, exame);
    data = readmatrix(sprintf('%d.csv', exame));
    resultado = [];
    for i = 1:size(data, 2)
        for j = i:size(data, 2)
            if i == j
                resultado = [resultado, zeros(1, 16)];
            else
                vetor_i = data(:, i)';
                vetor_j = data(:, j)';
                matriz = [vetor_i; vetor_j];

                if ~any(vetor_i(~isnan(vetor_i))) || ~any(vetor_j(~isnan(vetor_j)))
                    resultado = [resultado, zeros(1, 16)];
                else
                    atoms = PhiIDFull(matriz);
                    resultado = [resultado, ...
                        atoms.rtr, atoms.rtx, atoms.rty, atoms.rts, ...
                        atoms.xtr, atoms.xtx, atoms.xty, atoms.xts, ...
                        atoms.ytr, atoms.ytx, atoms.yty, atoms.yts, ...
                        atoms.str, atoms.stx, atoms.sty, atoms.sts];
                end
            end
        end
    end

    resultados(exame+1, :) = resultado;
end

% gerar os nomes das colunas
sufixos = {'rtr', 'rtx', 'rty', 'rts', 'xtr', 'xtx', 'xty', 'xts', ...
           'ytr', 'ytx', 'yty', 'yts', 'str', 'stx', 'sty', 'sts'};
colNames = cell(1, num_cols);
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

% salvar resultados
resultados = array2table(resultados, 'VariableNames', colNames);
writetable(resultados, 'resultados.csv');
