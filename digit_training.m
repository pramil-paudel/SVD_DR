% svd_digit_train_batch.m
% Author: Pramil Paudel
% Description: Event-by-event SVD with explicit batch training and evaluation

clc; 
clear;

%% === Configuration ===
train_csv = 'R:/TAPIA-TAKAKI_GRP/ML_Scratch/train.csv';
batch_size = 100;
num_classes = 10;

%% === Load Training Data ===
fprintf('Loading training data from: %s\n', train_csv);
[X_train, y_train] = load_digit_csv(train_csv, num_classes);
true_labels = get_label(y_train);

fprintf('✅ Training data: %d samples, %d features\n', size(X_train,1), size(X_train,2));

%% === Initialization ===
class_svds = struct();
class_counts = zeros(num_classes, 1);
n_total = size(X_train, 1);

%% === Shuffle data ===
shuffle_idx = randperm(n_total);
X_train = X_train(shuffle_idx, :);
true_labels = true_labels(shuffle_idx);

%% === Load Test Data Early for Evaluation ===
test_feature_file = 'R:/TAPIA-TAKAKI_GRP/ML_Scratch/test.csv';
test_label_file = 'R:/TAPIA-TAKAKI_GRP/ML_Scratch/mnist_submission.csv';
test_features = readmatrix(test_feature_file);
if size(test_features, 2) == 783
    warning('⚠️ Test data has 783 features — padding 1 zero column.');
    test_features(:, end+1) = 0;
end
X_test = double(test_features) / 255.0;
submission = readtable(test_label_file);
true_test_labels = submission.Label;
if length(true_test_labels) ~= size(X_test, 1)
    error('❌ Mismatch between test data and submission labels!');
end

%% === Process in batches ===
fprintf('\n🚀 Starting batch-based SVD training...\n');
total_batches = ceil(n_total / batch_size);
total_updates = 0;

for batch_num = 1:total_batches
    batch_start = (batch_num - 1) * batch_size + 1;
    batch_end = min(batch_start + batch_size - 1, n_total);
    X_batch = X_train(batch_start:batch_end, :);
    y_batch = true_labels(batch_start:batch_end);

    if ~isempty(fieldnames(class_svds))
        [pred_train, ~] = classify_with_svd(class_svds, X_train);
        train_acc = mean(pred_train == true_labels);
        [pred_test, ~] = classify_with_svd(class_svds, X_test);
        test_acc = mean(pred_test == true_test_labels);
    else
        train_acc = 0.0;
        test_acc = 0.0;
    end

    updated_count = 0;

    for i = 1:size(X_batch,1)
        x = X_batch(i, :)';
        label = y_batch(i);

        [pred, ~] = classify_with_svd(class_svds, x');
        if pred(1) == label
            continue;
        end

        if class_counts(label+1) == 0
            [U, S, V] = svd(x, 'econ');
        else
            [U, S, V] = event_by_event_svd(class_svds(label+1).U, class_svds(label+1).S, class_svds(label+1).V, x);
        end

        class_svds(label+1).U = U;
        class_svds(label+1).S = S;
        class_svds(label+1).V = V;
        class_counts(label+1) = class_counts(label+1) + 1;
        updated_count = updated_count + 1;
    end

    total_updates = total_updates + updated_count;
    update_ratio = 100 * total_updates / n_total;

    fprintf('Batch %d/%d | Size:%d | Train Sample Accuracy Before: %.2f%% | Test Sample Accuracy Before: %.2f%% | Updates on this batch: %d / %d | Data used for update: %.2f%%%%\n', ...
        batch_num, total_batches, batch_end - batch_start + 1, train_acc * 100, test_acc * 100, updated_count, batch_end - batch_start + 1, update_ratio);
end

%% === Final Accuracy ===
[pred_train, ~] = classify_with_svd(class_svds, X_train);
train_acc = mean(pred_train == true_labels);
fprintf('\n✅ Final Training Accuracy: %.2f%%\n', train_acc*100);

%% === Classify Test Samples ===
fprintf('\n Classifying test samples...\n');
[pred_test_labels, test_scores] = classify_with_svd(class_svds, X_test);
test_accuracy = mean(pred_test_labels == true_test_labels);
fprintf('🏂 Test Accuracy: %.2f%%\n', test_accuracy * 100);

fprintf('Generating ROC curves for test set...\n');
figure; hold on;

for c = 0:num_classes-1
    class_mask = (true_test_labels == c);
    num_positive = sum(class_mask);
    num_negative = sum(~class_mask);

    if num_positive == 0 || num_negative == 0
        fprintf('⚠️ Skipping ROC for class %d (only one class present)\n', c);
        continue;
    end

    [Xroc, Yroc, ~, AUC] = perfcurve(class_mask, test_scores(:, c+1), true);
    plot(Xroc, Yroc, 'DisplayName', sprintf('Digit %d (AUC = %.2f)', c, AUC));
end

xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curves - Test Set');
legend('show'); grid on;

%% === Helper Functions ===
function [X, y_onehot] = load_digit_csv(filepath, num_classes)
    opts = detectImportOptions(filepath);
    data = readmatrix(filepath, opts);
    y = data(:, 1);
    X = data(:, 2:end);
    X = double(X) / 255.0;
    y_onehot = one_hot_encode(y, num_classes);
end

function y_encoded = one_hot_encode(labels, num_classes)
    n = length(labels);
    y_encoded = zeros(n, num_classes);
    for i = 1:n
        y_encoded(i, labels(i)+1) = 1;
    end
end

function labels = get_label(y_onehot)
    [~, labels] = max(y_onehot, [], 2);
    labels = labels - 1;
end

function [predicted_labels, scores] = classify_with_svd(class_svds, X)
    n = size(X, 1);
    num_classes = 10;
    scores = zeros(n, num_classes);
    predicted_labels = zeros(n, 1);

    for i = 1:n
        x = X(i, :)';
        for c = 1:num_classes
            if length(class_svds) < c || ~isfield(class_svds(c), 'U') || isempty(class_svds(c).U)
                scores(i, c) = -inf;
                continue;
            end
            Uc = class_svds(c).U;
            proj = Uc * (Uc' * x);
            residual = norm(x - proj);
            scores(i, c) = -residual;
        end
        [~, predicted_labels(i)] = max(scores(i, :));
        predicted_labels(i) = predicted_labels(i) - 1;
    end
end

function [U_new, S_new, V_new, did_update] = event_by_event_svd(U, S, V, A)
    m = U' * A;
    p = A - U * m;
    P = orth(p);
    if isempty(P)
        U_new = U; S_new = S; V_new = V; did_update = false;
        return;
    end
    Ra = P' * p;
    z = zeros(size(m));
    K = [S, m; z', Ra];
    [tU, tSvec, tV] = svd(K, 'econ');
    r = size(S, 1);
    tU = tU(:, 1:r); tS = diag(tSvec(1:r)); tV = tV(:, 1:r);
    U_new = [U, P] * tU;
    V_top = V * tV(1:size(V,2), :);
    V_bottom = tV(size(V,2)+1:end, :);
    V_new = [V_top; V_bottom];
    S_new = diag(tS);
    did_update = true;
end
