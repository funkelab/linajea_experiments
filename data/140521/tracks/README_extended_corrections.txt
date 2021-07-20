Record keeping for corrections I (Caroline) made to extended tracks on 7.23.19

1) Error detected by collect_tracks
When I ran collect_tracks.py on the new extended ground truth, I got the following warning:
"track does not form a tree in time, cell 329 has (at least) parents 332 and 330"
Based on examination of TB_Extended.shifted.TrackingMatrixGT.csv, it appears that
the linage looks like this:

t
63              332
                 /
64          330 /
             | /  
65          329 

I corrected it so that it looks like this in a new file called 
TB_Extended_corrected.shifted.TrachingMatrixGT.csv:

t
63          332
             |   
64          330 
             |   
65          329 

Then I reran collect_tracks.py to create a file called extended_tracks_uncorrected.txt
and ran check_ground_truth.py to detect potential errors, and made the 
changes from (2)-(4) to create a file called extended_tracks_no_singletons_or_skips.txt.

2) Removed all unattached nodes (nodes with no edges).

Kate said that MaMuT hides these from her view, so she can never
delete them after she accidentally makes them. Each of them had a track
id of -1 in the extended_tracks.txt file. Here is the list of ids:

INFO:__main__:Found 101 unattached nodes: [11382, 10174, 13760, 9799, 9801, 9802, 9803, 9804, 9805, 9806, 9807, 9808, 9809, 9810, 9811, 2537, 2535, 2536, 2456, 2445, 2444, 2448, 2443, 2442, 2441, 2440, 2439, 2438, 2437, 2436, 2435, 2434, 2433, 2432, 2431, 2430, 2429, 2428, 2427, 2426, 2425, 2424, 2423, 2422, 2421, 2420, 2419, 2418, 2417, 2416, 2415, 2414, 2413, 2412, 2411, 2410, 2409, 2408, 2407, 2406, 2405, 9841, 2404, 2403, 2402, 2401, 2400, 2399, 2398, 2397, 2396, 2395, 2394, 2393, 2392, 3809, 3773, 24233, 24234, 24235, 24236, 24237, 79943, 79942, 79941, 249074, 248699, 248701, 248702, 248703, 248704, 248705, 248706, 248707, 248708, 248709, 248710, 248711, 241341, 242709, 242673]

3) Removed an incorrect edge detected by its exceedingly long length.

The script that checks ground truth for consistency detected this potentially incorrect edge by its length:
Max distance: 150.92881964687825 source {'position': (529, 3023.8, 916.51, 967.05), 'id': 11384} target {'position': (528, 3169.7, 929.32, 1003.5), 'id': 11386}

I examined this edge in MaMuT and determined that it was incorrect. The track
"split" at the 528 time step, with one correct branch and one incorrect brnach. 
Since this was right at the end of the video,
I just deleted the incorrect edge and the two following nodes before the
end of the video.

Here are the IDs of the nodes that I deleted:
11384
17631
17632

4) Fixed three edges that spanned two frames.

The script detected four edges that spanned two frames:
________INVALID EDGE SKIPS FRAMES______________
Edge: {'source': 11185, 'target': 11184, 'distance': 18.591473852279737}
Source: {'position': (259, 2765.9, 1052.6, 862.67), 'id': 11185}
Target: {'position': (257, 2748.4, 1057.2, 858.4), 'id': 11184}
________INVALID EDGE SKIPS FRAMES______________
Edge: {'source': 4252, 'target': 4251, 'distance': 22.758534662846987}
Source: {'position': (427, 3013.8, 405.71, 1150.9), 'id': 4252}
Target: {'position': (425, 2993.5, 412.18, 1142.9), 'id': 4251}
________INVALID EDGE SKIPS FRAMES______________
Edge: {'source': 188367, 'target': 187423, 'distance': 25.69746195697594}
Source: {'position': (410, 1921.434588, 816.380455, 865.837109), 'id': 188367}
Target: {'position': (408, 1904.658262, 803.939093, 880.807979), 'id': 187423}
________INVALID EDGE SKIPS FRAMES______________
Edge: {'source': 255257, 'target': 255256, 'distance': 19.93890400278212}
Source: {'position': (340, 3205.442921, 683.595777, 1141.989821), 'id': 255257}
Target: {'position': (338, 3186.97302, 679.005699, 1136.043912), 'id': 255256}

I looked at all of these in MaMuT and the first three were almost identical situations.
All of them were at a division, and one actually should have pointed to the "middle" 
time point that was skipped. For example, the first one looked like this in track scheme
 before correction:

t
257          11184
             /  \
258        11183 \
             |    \
259        11182 11185

and this after correction:

t
257          11184
               |  
258          11183 
             /    \
259        11182 11185

I corrected all three of these by changing the parent_id of the source node to the 
child of the target.

The fourth was not at a division, and simply was a track that skipped an edge like this:

t
338         255256
              |
339           |
              |
340         255257

I looked at this in MaMuT and it is actually the same cell, it is just missing a point in the middle.
This missing point is basically in the same place as the cell in 340. I added this point
so that it looks like this:

t
338         255256
              |
339         256307
              |
340         255257


Then I uploaded the corrected ground truth into the mongo database so that I could check for
duplicate tracks using the check_duplicate_gt_tracks.py with a cell radius of 10 and a threshold
of 15 overlapping cells. Since these tracks are compiled from multiple sources of annotations,
it's possible that the same track will be annotated twice. Although this doesn't matter SO much
for training, it would mess up the evaluation.

5) Remove duplicate tracks

Here are the results of running the script:

{(4, 5): 675, (5, 18): 463, (12, 27): 90, (27, 51): 160, (30, 59): 1, (13, 35): 2, (19, 100): 17, (100, 118): 2, (52, 83): 1, (78, 88): 3, (7, 151): 1, (85, 159): 3, (72, 104): 2, (82, 161): 2}
Tracks 4 and 5 are duplicates
4 has 717 cells, 5 has 1206 cells
Would delete track 4
Tracks 5 and 18 are duplicates
5 has 1206 cells, 18 has 485 cells
Would delete track 18
Tracks 12 and 27 are duplicates
12 has 230 cells, 27 has 387 cells
Would delete track 12
Tracks 27 and 51 are duplicates
27 has 387 cells, 51 has 160 cells
Would delete track 51
Tracks 19 and 100 are duplicates
19 has 171 cells, 100 has 160 cells
Would delete track 100

I looked at these all in MaMuT and here were my thoughts one by one:

Tracks 4 and 5
Here is the first pair of cells where tracks 4 and 5 overlap:
INFO:__main__:These cells are close together: 18814 {'t': 48, 'z': 2464.4, 'y': 976.49, 'x': 923.63, 'score': 0, 'track_id': 4}    231924 {'t': 48, 'z': 2466.628561, 'y': 975.553336, 'x': 921.950177, 'score': 0, 'track_id': 5}

Track 4 starts at time point 37. Track 5 starts at time point 48. 
Beyond time step 48, however, track 5 has every single cell that track 4 has,
and the tracks are extended further.

SOLUTION: Put the beginning of track 4 onto track 5, then delete the rest of track 4.
        tr4        tr5 
46      18816
          |
47      18815 
          |x   \(new)
48      18814x    231924
          |x        |
49      18813x    231923
          |x        |
50      18812x    231922 


Tracks 5 and 18
Here is the first pair of cells where tracks 5 and 18 overlap:
INFO:__main__:These cells are close together: 18719 {'t': 139, 'z': 2671.7, 'y': 924.99, 'x': 938.9, 'score': 0, 'track_id': 18}    83833 {'t': 139, 'z': 2672.390921, 'y': 923.861597, 'x': 937.580003, 'score': 0, 'track_id': 5}

Track 5 starts at time piont 48, and 18 starts at 139.
They are almost identical from time 139 on, except at the end of one of the branches,
where they disagree about the which cell to follow. They start diverging at frame 386, 
where track 5 continues in one direction until frame 390, and track 18 continues
in another direction until frame 401.

SOLUTION: Delete track 18 (first ID: 18719), and end the controversial branch of 5 at frame 385
(delete including and after ID 87916) since it's unclear which track is correct.


Tracks 12 and 27 and 51
Here is the first pair of cells where 12 and 27 overlap
INFO:__main__:These cells are close together: 255087 {'t': 195, 'z': 2382.763634, 'y': 685.738262, 'x': 902.839579, 'score': 0, 'track_id': 12}    24382 {'t': 195, 'z': 2392.5, 'y': 685.2, 'x': 901.4, 'score': 0, 'track_id': 27}
Here is the first pair of cells where 27 and 51 overlap
INFO:__main__:These cells are close together: 24334 {'t': 242, 'z': 2667.0, 'y': 635.68, 'x': 950.67, 'score': 0, 'track_id': 27}    255038 {'t': 242, 'z': 2668.972429, 'y': 635.06422, 'x': 947.345354, 'score': 0, 'track_id': 51}

I'm dealing with these tracks as a trio. Track 12 starts at 128, 27 starts at 194, and 51 starts at 242.

Essentially, tracks 12 and 51 trace different branches of track 27. 
12 and 27 trace the same track from 194 to 241, with some slight deviations in the early section
due to noise in the data. Between 241 and 242, track 27 splits, and 12 follows one of the branches,
and 51 starts at 252 and follows the other. 

Track 51 ends at frame 401 and the matching branch of 27 continues all the way to the end (531).
So track 27 contains all the information that 51 does. 

Track 12 continues to frame 357, and the branch of 27 ends at 290.

SOLUTION: delete track 51 (first ID: 255038). Append the beginning of 12 (128-193) and
the end of 12 (291-357) to 27 and delete the middle.
Track 12 has no divisions so we don't have to worry about those when deleting.
        tr12        tr27 
193    255089   
         |x    \(new)
194    255088x     24383 
         |x          |
195    255087x     24382    
         .           .
         .           .
         .           .
289    255207x     24428
         |x          | 
290    255208x     24429    
         |x     /(new)
291    255209   


Tracks 19 and 100
Here is the first pair of cells where 19 and 100 overlap:
INFO:__main__:These cells are close together: 24049 {'t': 318, 'z': 3025.9, 'y': 679.02, 'x': 1061.4, 'score': 0, 'track_id': 100}    251624 {'t': 318, 'z': 3031.180658, 'y': 677.974176, 'x': 1064.707415, 'score': 0, 'track_id': 19}

Track 19 starts at frame 164 and ends at frame 334. Track 100 starts at frame 317 and ends at frame 476. From frames 317 to 334, the tracks are the same.

SOLUTION: Append the end of track 100 (335-476) to track 19. Delete the beginning of track 100 (317-334).
        tr19        tr100 
334    251608       24065x
              \(new)  |x
335                 24066 
                      |
336                 24067

It was too hard to make all these corrections by hand, so I wrote a class that can edit the tracks.txt.
I then used the script remove_duplicate_tracks.py to make the changes described above and write
the deduplicated tracks to extended_tracks_deduplicated.py


6) MaMuT Inspection

Finally I loaded these tracks into MaMuT. Based on inspecting the tracks,
it looked like there were a fair number of missing edges. I started to identify
them by hand, but it took too long, so I wrote a script that for each cell without
a parent, identified any potential parents within a certain radius. I then checked all
90 of these out by hand in MaMuT and compiled a list of edges that needed to be added. 
I then used the class from the deduplication step and the script add_missing_edges.py
to create these edges and merge the tracks,
creating a new file called extended_tracks_add_missing_edges.txt.

Here are a list of the edges I added: 
Radius=20
Approved 75 missing edges: [(232141, 232143), (252954, 251773), (79976, 79978), (15055, 15056), (79954, 79957), (90080, 90081), (256284, 256286), (252244, 252251), (11214, 11216), (249771, 249772), (1152, 1154), (12831, 12832), (1426, 10581), (4739, 4740), (250191, 250192), (247295, 247296), (12498, 11622), (20168, 20169), (24046, 89977), (4409, 4410), (10041, 10043), (254485, 254487), (239767, 239769), (16458, 16460), (254091, 254088), (238699, 238701), (13457, 13458), (253036, 253034), (15266, 15269), (10011, 10013), (249952, 249953), (10008, 10009), (4357, 4358), (12445, 12446), (251963, 247044), (3093, 3094), (12719, 12726), (13037, 13038), (245294, 245295), (92659, 92743), (9969, 9970), (246869, 246870), (252067, 252069), (13207, 13209), (4261, 4262), (245714, 245715), (245249, 245261), (245241, 245244), (1558, 1559), (11506, 11508), (251165, 251167), (251502, 251503), (239653, 239654), (246816, 246817), (14773, 14774), (254363, 254364), (245682, 245685), (242487, 242488), (250884, 250885), (239612, 239613), (2986, 2987), (242478, 242479), (15125, 15127), (13084, 13085), (923, 925), (245881, 245882), (16283, 16284), (3232, 3233), (12896, 12897), (242697, 249476), (242723, 242725), (9016, 1181), (4938, 4939), (251791, 251792), (247661, 247662)]
Radius=30
Approved 7 missing edges: [(14997, 15001), (5229, 5231), (1703, 1438), (10061, 10064), (9930, 10123), (245384, 242221), (9839, 9840)]

