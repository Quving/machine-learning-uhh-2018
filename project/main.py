#!/usr/bin/env python

from globalterrorismdb_parser import GlobalTerrorismDBParser

if __name__ == "__main__":
    gt_parser = GlobalTerrorismDBParser()

    # Plot histogram for country
    gt_parser.plot_histogram_for_column("country", bins=250, xlabel="Country Id", ylabel="Frequency")

    # Plot geographic map
    gt_parser.plot_geographical_heatmap(filename="geographic_map.png")
