import linajea
import linajea.tracking
import linajea.evaluation
import daisy
import logging
import argparse
import numpy as np
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MatchedEdgesWriter:
    def __init__(
            self,
            db_name,
            mongo_url,
            tgmm_db_name,
            frames,
            parameters_id,
            gt_db_name,
            basedir,
            matching_threshold=15):
        self.our_track_graph = self._get_tracks(
                db_name, mongo_url, frames=frames, parameters_id=parameters_id)
        self.greedy_track_graph = self._get_tracks(
                db_name, mongo_url, frames=frames, parameters_id='selected_greedy')
        self.tgmm_track_graph = self._get_tracks(
                tgmm_db_name, mongo_url, frames=frames, parameters_id=None)
        self.gt_track_graph = self._get_tracks(
                gt_db_name, mongo_url, frames=frames, parameters_id=None)

        self.frames = frames if frames is not None\
            else self.gt_track_graph.get_frames()
        self.num_frames = frames[1] - frames[0]
        self.matching_threshold = matching_threshold
        self.basedir = basedir
        self.gt_tracks = self.gt_track_graph.get_tracks()
        self.gt_tracks = [t for t in self.gt_tracks
                          if t.number_of_nodes() >= self.num_frames]

        self.our_matched_edges = None
        self.greedy_matched_edges = None
        self.tgmm_matched_edges = None
        self.__match_track_graphs()

    def write_matches_by_node(self, gt_node, out_dir):
        # write the matches for the gt_track with the given node
        logger.info("Finding gt track with node %d" % gt_node)
        for index, gt_track in enumerate(self.gt_tracks):
            selected = None
            if gt_node in gt_track:
                logger.debug("Found it!")
                selected = index
                break
        if selected is None:
            logger.error("No gt track found with node %d" % gt_node)
            return
        self._write_matches_for_track(selected)

    def write_median_fn_tracks(self):
        logger.info("Finding tracks with median FNs in our model")
        fns = []
        for index, gt_track in enumerate(self.gt_tracks):
            gt_edges = gt_track.number_of_edges()
            our_edges = len(self.our_matched_edges[index])
            fns.append(gt_edges - our_edges)
        median_fns = int(np.median(fns))
        logger.info("Median fns: %d" % median_fns)
        indexes = np.where(np.array(fns) == median_fns)[0]
        for selected in indexes:
            self._write_matches_for_track(selected)

    def _write_matches_for_track(self, index, fps=True):
        out_dir = os.path.join(self.basedir, str(index))
        os.makedirs(out_dir, exist_ok=True)
        selected_track = self.gt_tracks[index]
        our_edges = self.our_matched_edges[index]
        tgmm_edges = self.tgmm_matched_edges[index]
        greedy_edges = self.greedy_matched_edges[index]
        self._write_tracks(selected_track,
                           os.path.join(out_dir, 'gt_%s.csv'))
        self._write_tracks(self.our_track_graph,
                           os.path.join(out_dir, 'ours_%s.csv'),
                           matched_edges=our_edges)
        self._write_tracks(self.greedy_track_graph,
                           os.path.join(out_dir, 'greedy_%s.csv'),
                           matched_edges=greedy_edges)
        self._write_tracks(self.tgmm_track_graph,
                           os.path.join(out_dir, 'tgmm_%s.csv'),
                           matched_edges=tgmm_edges)
        if fps:
            our_fps = self._get_fp_divs(
                self.our_track_graph, our_edges)
            greedy_fps = self._get_fp_divs(
                self.greedy_track_graph, greedy_edges)
            tgmm_fps = self._get_fp_divs(
                self.tgmm_track_graph, tgmm_edges)
            self._write_tracks(self.our_track_graph,
                               os.path.join(out_dir, 'ours_fps_%s.csv'),
                               matched_edges=our_fps)
            self._write_tracks(self.greedy_track_graph,
                               os.path.join(out_dir, 'greedy_fps_%s.csv'),
                               matched_edges=greedy_fps)
            self._write_tracks(self.tgmm_track_graph,
                               os.path.join(out_dir, 'tgmm_fps_%s.csv'),
                               matched_edges=tgmm_fps)

    def _write_tracks(self, track_graph, out_file, matched_edges=None):
        if matched_edges is not None:
            if len(matched_edges) == 0:
                return
            track_graph = track_graph.edge_subgraph(matched_edges)
        tracks = track_graph.get_tracks()
        split_tracks = self._split_divisions(tracks)
        for index, track in enumerate(split_tracks):
            self._write_track_to_csv(
                    track,
                    out_file % index)

    def _write_track_to_csv(
            self,
            track,
            out_file):
        remove_nodes = []
        for node, data in track.nodes(data=True):
            if 't' not in data:
                remove_nodes.append(node)
        track.remove_nodes_from(remove_nodes)

        fields = ['t', 'z', 'y', 'x']
        delim = ', '
        # (world coordiantes)
        with open(out_file, 'w') as f:
            # write header
            f.write(delim.join(fields))
            f.write('\n')

            locations = []
            for node_id, data in track.nodes(data=True):
                t = data['t']
                z = data['z']
                y = data['y']
                x = data['x']

                locations.append([
                    t, z, y, x])
            locations = sorted(locations)
            to_write = []
            for loc in locations:
                to_write.append(delim.join(list(map(str, loc))))
            f.write('\n'.join(to_write))

    def _get_tracks(
            self,
            db_name,
            mongo_url,
            frames=None,
            parameters_id=None):
        if frames is not None:
            start_frame, end_frame = frames
            assert(start_frame < end_frame)
            num_frames = end_frame - start_frame
        else:
            start_frame = None
            num_frames = None
        roi = daisy.Roi((start_frame, 0, 0, 0), (num_frames, 1e10, 1e10, 1e10))

        try:
            parameters_id = int(parameters_id)
        except:
            pass

        if type(parameters_id) == str:
            db = linajea.CandidateDatabase(
                    db_name, mongo_url)
            db.selected_key = parameters_id
        else:
            db = linajea.CandidateDatabase(
                    db_name, mongo_url, parameters_id=parameters_id)
        logger.info("Reading cells and edges in %s" % roi)
        if parameters_id is None:
            subgraph = db[roi]
        else:
            subgraph = db.get_selected_graph(
                    roi, edge_attrs=['prediction_distance'])
        track_graph = linajea.tracking.TrackGraph(subgraph, frame_key='t')
        logger.info("Found %d nodes and %d edges in db %s with key %s"
                    % (track_graph.number_of_nodes(),
                       track_graph.number_of_edges(),
                       db_name,
                       str(parameters_id)))
        return track_graph

    def _get_fp_divs(self, track_graph, matched_edges):
        in_edges = [list(track_graph.in_edges(target))
                    for source, target in matched_edges]
        fp_divs = []
        potential_fp_div_edges = [e for e in in_edges if len(e) > 1]
        for edges in potential_fp_div_edges:
            for edge in edges:
                if edge not in matched_edges:
                    fp_divs.append(edge)
        if len(fp_divs) > 0:
            print("divisions: %d" % len(potential_fp_div_edges))
            print("Len in_edges: %d" % len(in_edges))
            print("first entry in in_edges: %s" % str(in_edges[0]))
            print("fp div edges: %d" % len(fp_divs))
        return fp_divs

    def _get_matched_edges(
            self,
            gt_tracks,
            gt_edges,
            rec_edges,
            edge_matches,
            fps=True):
        gt_edge_to_rec = {gt_edges[gt]: rec_edges[rec]
                          for gt, rec in edge_matches}
        gt_track_to_edges = {}
        for index, gt_track in enumerate(gt_tracks):
            gt_track_to_edges[index] = []
            for gt_edge in gt_track.edges():
                if gt_edge in gt_edge_to_rec:
                    gt_track_to_edges[index].append(gt_edge_to_rec[gt_edge])

        return gt_track_to_edges

    def _split_divisions(self, tracks):
        split_tracks = []
        for track in tracks:
            parents = [n for n in track.nodes() if track.in_degree(n) > 1]
            for parent in parents:
                # create dummy node
                dummy = 1
                while dummy in track.nodes():
                    dummy += 1
                data = track.nodes[parent]
                track.add_node(dummy, **data)
                in_edges = list(track.in_edges(parent))
                assert len(in_edges) == 2
                replace_edge = in_edges[0]
                track.add_edge(replace_edge[0], dummy)
                track.remove_edge(*replace_edge)
            split_tracks.extend(track.get_tracks())
        return split_tracks

    def __match_track_graphs(self):
        gt_edges, rec_edges, edge_matches, edge_fps =\
            linajea.evaluation.match_edges(
                self.gt_track_graph,
                self.our_track_graph,
                matching_threshold=self.matching_threshold)

        _, greedy_edges, greedy_edge_matches, greedy_edge_fps =\
            linajea.evaluation.match_edges(
                self.gt_track_graph,
                self.greedy_track_graph,
                matching_threshold=self.matching_threshold)

        _, tgmm_edges, tgmm_edge_matches, tgmm_edge_fps =\
            linajea.evaluation.match_edges(
                self.gt_track_graph,
                self.tgmm_track_graph,
                matching_threshold=self.matching_threshold)

        # get matched edges per track
        self.our_matched_edges = self._get_matched_edges(
                self.gt_tracks, gt_edges, rec_edges, edge_matches)
        self.greedy_matched_edges = self._get_matched_edges(
                self.gt_tracks, gt_edges, greedy_edges, greedy_edge_matches)
        self.tgmm_matched_edges = self._get_matched_edges(
                self.gt_tracks, gt_edges, tgmm_edges, tgmm_edge_matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('gt_db_name')
    parser.add_argument('tgmm_db_name')
    parser.add_argument('mongo_url')
    parser.add_argument('out_dir')
    parser.add_argument('-f', '--frames', type=int, nargs=2, default=None)
    parser.add_argument('-id', '--parameters_id')
    parser.add_argument('-n', '--gt_node', type=int, default=None)
    parser.add_argument('-t', '--threshold', default=15, type=int)
    args = parser.parse_args()
    writer = MatchedEdgesWriter(
            args.db_name,
            args.mongo_url,
            args.tgmm_db_name,
            args.frames,
            args.parameters_id,
            args.gt_db_name,
            args.out_dir,
            matching_threshold=args.threshold)
    if args.gt_node:
        writer.write_matches_by_node(args.gt_node)
    else:
        writer.write_median_fn_tracks()
