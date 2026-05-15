function T_balanced = balance_dataset(T)

classes = unique(T.attack_class);
classes(classes == 0) = [];

counts = arrayfun(@(c) sum(T.attack_class == c), classes);
min_count = min(counts);

T_balanced = [];

for c = classes'
    idx = find(T.attack_class == c);
    idx = idx(randperm(length(idx), min_count));
    T_balanced = [T_balanced; T(idx,:)];
end

% balance normal
normal_idx = find(T.attack_class == 0);
normal_idx = normal_idx(randperm(length(normal_idx), min_count * length(classes)));

T_balanced = [T_balanced; T(normal_idx,:)];

% shuffle
T_balanced = T_balanced(randperm(height(T_balanced)),:);

end