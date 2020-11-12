from time import time
from V_Visualization import visualization
from B_Analysis_FC import analyzer


if __name__ == "__main__":
	start = time()
	analyz = analyzer()
	viz = visualization()
	
	#analyz.calc_mean_edge()
	#viz.plot_mean_edge()

	analyz.excl_outliers()

	#analyz.stack_fcs()
	#viz.plot_edge_dist()
	
	#analyz.calc_group_mean_fc()
	#viz.plot_group_mean_fc()
	

	#viz.plot_cross_hemi_corr()
	
	#analyz.calc_GBC()
	#viz.plot_avg_GBC_diff()
	#viz.plot_avg_GBC()
	
	analyz.calc_nbs()
	#analyz.dabest_avg_GBC()
	#analyz.dabest_region_GBC()
	
	#analyz.calc_net_measures()
	#viz.plot_net_measures()
	#analyz.dabest_net_measures()
	end = time()
	print('Time: ', end-start)
