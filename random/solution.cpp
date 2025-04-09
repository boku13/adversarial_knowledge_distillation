#include <bits/stdc++.h>
using namespace std;

int getMinRepairCost(int g_nodes, int g_edges, vector<int> g_from, vector<int> g_to, vector<int> g_weight, int k) {
    // Create adjacency list with weights
    vector<vector<pair<int, int>>> adj(g_nodes + 1);
    for(int i = 0; i < g_edges; i++) {
        adj[g_from[i]].push_back({g_to[i], g_weight[i]});
        adj[g_to[i]].push_back({g_from[i], g_weight[i]});
    }
    
    // Binary search on the minimum cost
    long long left = 0, right = 1e9;
    int ans = -1;
    
    while(left <= right) {
        long long mid = (left + right) / 2;
        
        // BFS to check if we can reach node g_nodes using at most k edges
        // and roads with cost <= mid
        vector<int> dist(g_nodes + 1, INT_MAX);
        queue<pair<int, int>> q; // {node, distance}
        q.push({1, 0});
        dist[1] = 0;
        
        bool possible = false;
        while(!q.empty()) {
            int node = q.front().first;
            int d = q.front().second;
            q.pop();
            
            if(node == g_nodes) {
                possible = true;
                break;
            }
            
            if(d >= k) continue;
            
            for(auto [next, weight] : adj[node]) {
                if(weight <= mid && dist[next] > d + 1) {
                    dist[next] = d + 1;
                    q.push({next, d + 1});
                }
            }
        }
        
        if(possible) {
            ans = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return ans;
}

// For testing
int main() {
    int g_nodes = 5;
    int g_edges = 6;
    vector<int> g_from = {1, 3, 4, 3, 1, 2};
    vector<int> g_to = {3, 4, 5, 5, 2, 5};
    vector<int> g_weight = {2, 4, 6, 9, 7, 8};
    int k = 2;
    
    cout << getMinRepairCost(g_nodes, g_edges, g_from, g_to, g_weight, k) << endl;
    return 0;
} 