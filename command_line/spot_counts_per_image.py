from __future__ import absolute_import, division, print_function

import json
import sys

import iotbx.phil
from dials.util import tabulate
from dials.array_family import flex
from dials.util.options import OptionParser
from dials.util.options import flatten_reflections, flatten_experiments
from dials.algorithms.spot_finding import per_image_analysis

help_message = """

Reports the number of strong spots and computes an estimate of the resolution
limit for each image, given the results of dials.find_spots. Optionally
generates a plot of the per-image statistics (plot=image.png).

Examples::

  dials.spot_counts_per_image imported.expt strong.refl

  dials.spot_counts_per_image imported.expt strong.refl plot=per_image.png
"""

phil_scope = iotbx.phil.parse(
    """\
resolution_analysis = True
  .type = bool
plot = None
  .type = path
json = None
  .type = path
split_json = False
  .type = bool
joint_json = True
  .type = bool
individual_plots = False
  .type = bool
id = None
  .type = int(value_min=0)
"""
)


def print_remerged_table(imageset_numbers, image_numbers, stats):
    """Generates and prints a PIA table with inserted imageset information"""

    rows = per_image_analysis.table(stats)

    # Edit this table to insert the imageset/image number columns
    rows[0] = ["Imageset", "IImage"] + rows[0]
    for i in range(1, len(rows)):
        rows[i] = [imageset_numbers[i - 1], image_numbers[i - 1]] + rows[i]

    print(tabulate(rows, headers="firstrow"))


def run(args):
    usage = "dials.spot_counts_per_image [options] imported.expt strong.refl"

    parser = OptionParser(
        usage=usage,
        read_reflections=True,
        read_experiments=True,
        phil=phil_scope,
        check_format=False,
        epilog=help_message,
    )

    params, options = parser.parse_args(show_diff_phil=False)
    reflections = flatten_reflections(params.input.reflections)
    experiments = flatten_experiments(params.input.experiments)

    if not reflections and not experiments:
        parser.print_help()
        return

    # FIXME may want to change this to allow many to be passed i.e.
    # from parallel runs
    if len(reflections) != 1:
        sys.exit("Only one reflection list may be passed")
    reflections = reflections[0]
    expts = set(reflections["id"])
    if len(expts) and max(expts) >= len(experiments.imagesets()):
        sys.exit("Unknown experiments in reflection list")

    reflections.centroid_px_to_mm(experiments)
    reflections.map_centroids_to_reciprocal_space(experiments)
    imageset_expts = {}
    imageset_scans = {}
    for j, experiment in enumerate(experiments):
        imageset = experiment.imageset
        if imageset in imageset_expts:
            imageset_expts[imageset].append(j)
            imageset_scans[imageset] += experiment.scan
        else:
            imageset_expts[imageset] = [j]
            imageset_scans[imageset] = experiment.scan

    if params.id is not None:
        reflections = reflections.select(reflections["id"] == params.id)

    all_stats = []
    for imageset in imageset_expts:
        imageset.set_scan(imageset_scans[imageset])

        selected = flex.bool(reflections.size(), False)
        for j in imageset_expts[imageset]:
            selected.set_selected(reflections["id"] == j, True)
        refl = reflections.select(selected)
        stats = per_image_analysis.stats_imageset(
            imageset,
            refl,
            resolution_analysis=params.resolution_analysis,
            plot=params.individual_plots,
        )
        all_stats.append(stats)

    # Store the imageset number and image number for every row
    summary_imageset_number = []
    summary_image_number = []

    # transpose stats
    summary_table = {attrib: [] for attrib in per_image_analysis.Stats._fields}
    for i, s in enumerate(all_stats):
        for attrib, value in s._asdict().items():
            summary_table[attrib].extend(value)
        # Make a boatload of assumptions about image/imageset number mapping
        summary_imageset_number.extend([i + 1] * len(value))
        summary_image_number.extend(list(range(1, len(value) + 1)))

    # # Because we don't want to change PIA code, reconstruct the imageset/image
    # # numbers from the known order and size of imagesets
    # for imageset in imageset_expts:
    print_remerged_table(
        summary_imageset_number,
        summary_image_number,
        per_image_analysis.Stats(**summary_table),
    )

    # FIXME this is now probably nonsense...
    overall_stats = per_image_analysis.stats_single_image(
        imageset, reflections, resolution_analysis=params.resolution_analysis
    )
    rows = [
        ("Overall statistics", ""),
        ("#spots", "%i" % overall_stats.n_spots_total),
        ("#spots_no_ice", "%i" % overall_stats.n_spots_no_ice),
        ("d_min", "%.2f" % overall_stats.estimated_d_min),
        (
            "d_min (distl method 1)",
            "%.2f (%.2f)"
            % (overall_stats.d_min_distl_method_1, overall_stats.noisiness_method_1),
        ),
        (
            "d_min (distl method 2)",
            "%.2f (%.2f)"
            % (overall_stats.d_min_distl_method_1, overall_stats.noisiness_method_1),
        ),
    ]
    print(tabulate(rows, headers="firstrow"))

    if params.json:
        if params.split_json:
            for k, v in stats._asdict().items():
                start, end = params.json.split(".")
                with open("%s_%s.%s" % (start, k, end), "w") as fp:
                    json.dump(v, fp)
        if params.joint_json:
            with open(params.json, "w") as fp:
                json.dump(stats._asdict(), fp)
    if params.plot:
        import matplotlib

        matplotlib.use("Agg")
        per_image_analysis.plot_stats(stats, filename=params.plot)


if __name__ == "__main__":
    run(sys.argv[1:])
