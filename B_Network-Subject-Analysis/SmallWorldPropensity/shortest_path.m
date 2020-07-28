function SP = shortest_path(M)
% Function to compute the shortest path between all nodes in the network
% using the Dijstrak Algorithm. 

% Input 	Adjacency Matrix (inversed edge values)
% Output	Shortest Path Matrix

SP = zeros(size(M));
num_nodes = size(M,1);

for n=1:num_nodes
	node_dist = inf(num_nodes, 1);
	node_dist(n) = 0; 
	unvisited = logical(ones(num_nodes,1));
	
	while any(unvisited) == 1 
        unvisited_nodes = node_dist(unvisited);
		min_value = min(unvisited_nodes);
		current = find(node_dist==min_value);
		unvisited(current) = 0;
		for k=1:num_nodes
			dist = node_dist(current) + M(current, k);
			if node_dist(k) > dist
				node_dist(k) = dist;
			end
		end
	end
	SP(:,n) = node_dist;
	SP(n,:) = node_dist;
end 
end 