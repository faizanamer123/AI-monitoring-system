"""
From the original EgoHands code from Indiana University-
get_meta_by  Returns EgoHands video metadata structures for videos which
match the argument filters. For a description of the metadata structure
see buildMetadata help.

  C = get_meta_by() returns metadata for all videos.

  C = get_meta_by(FilterName, Value, ...) returns metadat for all videos
  matching the filters. Possible filters and values are listed below:

  Filter            Possible Values        Info
  ----------        --------------------   ------------------------
  'Location'        'OFFICE','COURTYARD',  Video background location
                    'LIVINGROOM'

  'Activity'        'CHESS','JENGA',       Activity in video
                    'PUZZLE','CARDS'

  'Viewer'          'B','S','T','H'        Identity of egocentric viewer

  'Partner'         'B','S','T','H'        Identity of egocentric partner

  Multiple filters and values can be mixed, for example:
  get_meta_by('Location','OFFICE, COURTYARD', 'Activity','CHESS', 'Viewer','B, S, T')
  would return all videos of Chess played with B,S, or T as the egocentric
  observer filmed at the Office or Courtyard locations.
"""
import scipy.io as sio
import pandas as pd

def get_meta_by(*args):

    """
        :param location: OFFICE, COURTYARD, LIVINGROOM
        :param activity: CHESS, JENGA, PUZZLE, CARDS
        :param viewer: B, S, T, H
        :param partner: B, S, T, H
        :return: DataFrame of videos matching every filter given.

        Filters are passed as name/value pairs and may appear in any order.
        Values accept one or more comma-separated entries, with or without
        spaces: 'B,S' and 'B, S' behave identically.

        An unrecognised filter name raises ValueError rather than being
        silently ignored.
    """

    location_params = 'OFFICE, COURTYARD, LIVINGROOM'
    viewer_params = 'B, S, T, H'
    partner_params = 'B, S, T, H'
    activity_params = 'CHESS, JENGA, PUZZLE, CARDS'


    ## assigning from arguments given
    # Arguments are alternating name/value pairs. The previous loop tracked state
    # across iterations and silently dropped any name it did not recognise, so a
    # typo -- or the main_split filter the docstring used to promise -- returned the
    # full 48-video table with no indication the filter had been ignored.
    valid_filters = ("Location", "Activity", "Viewer", "Partner")
    if len(args) % 2:
        raise ValueError(
            f"get_meta_by expects FilterName/Value pairs; got {len(args)} arguments: {args}"
        )
    selected = {}
    for name, value in zip(args[::2], args[1::2]):
        if name not in valid_filters:
            raise ValueError(
                f"unknown filter {name!r}. Valid filters are: {', '.join(valid_filters)}"
            )
        selected[name] = value

    location_params = selected.get("Location", location_params)
    activity_params = selected.get("Activity", activity_params)
    viewer_params = selected.get("Viewer", viewer_params)
    partner_params = selected.get("Partner", partner_params)

    # splitting each input variable so we can check multiple conditions

    location = [v.strip() for v in location_params.split(",")]
    activity = [v.strip() for v in activity_params.split(",")]
    viewer = [v.strip() for v in viewer_params.split(",")]
    partner = [v.strip() for v in partner_params.split(",")]

    # loading metadata.mat
    meta_contents = sio.loadmat('./metadata.mat')
    annotations = meta_contents['video'][0]
    annotations_df = pd.DataFrame(annotations, columns=['video_id', 'partner_video_id',
                                                        'ego_viewer_id', 'partner_id',
                                                        'location_id', 'activity_id',
                                                        'labelled_frames'])


    # Data cleaning. Loading the .mat through numpy leaves every single-value cell
    # wrapped in a length-1 array, so unwrap them all.
    #
    # All four columns are cleaned BEFORE any filtering. The previous version
    # unwrapped one column, sliced, unwrapped the next, sliced again -- so the frame
    # handed back had ego_viewer_id as a str but partner_id, location_id and
    # activity_id still as ndarrays, and `row['location_id'] == 'OFFICE'` silently
    # compared an array to a string.
    for column in ('ego_viewer_id', 'partner_id', 'location_id', 'activity_id'):
        annotations_df[column] = annotations_df[column].apply(lambda cell: cell[0])

    queried_activities = annotations_df.loc[
        annotations_df['ego_viewer_id'].isin(viewer)
        & annotations_df['partner_id'].isin(partner)
        & annotations_df['location_id'].isin(location)
        & annotations_df['activity_id'].isin(activity)
    ]

    return queried_activities
