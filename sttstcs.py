import cv2
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict
from configs.colored import bcolors as bc


def approve(
        tresholds: Dict[str, float],
        face_confidence: float,
        face_class: str):
    """
    Check confidence

    Args:
        tresholds ([dict]): [ideal values]
        face_confidence ([float]): [current value]
        face_class ([str]): ['with_mask' or 'no_mask' class]

    Returns:
        [bool]
    """

    return face_confidence > tresholds[face_class]


def save_screenshot(**kwargs):
    """
    Save screenshot when any disturbance detected
    """
    frame = kwargs.get('frame', None)

    if kwargs['mode'] == 'd':
        path_to_save = kwargs.get(
            'path_to_contraventions_screenshots',
            'statistics/screenshots/distance/')
    elif kwargs['mode'] == 'm':
        path_to_save = kwargs.get(
                                'path_to_masks_screenshots', 
                                'statistics/screenshots/masks/')

    cv2.imwrite(path_to_save + 'frame.jpg', frame)


def write_data_to_file(**kwargs):
    """
    Append row to a csv file when has called

    Args:
        path_to_masks_statistics ([str]): [path to file where statistics saved]
        path_to_contraventions_statistics ([str]): [path to file where statistics saved]
        amount_people ([int]): [all detected peoples]
        amount_classes ([dict]): [dict with categories and tresholds]
        contraventions ([dict]): distance incidents
        absolute_time ([float]): [time between start of analizing algorythm and now]
    """
    amount_people = kwargs.get('amount_people')
    amount_classes = kwargs.get('amount_classes')
    contraventions = kwargs.get('contraventions')

    absolute_time = kwargs.get('absolute_time')
    path_to_masks_statistics = kwargs.get('path_to_masks_statistics')
    path_to_contraventions_statistics = kwargs.get('path_to_contraventions_statistics')

    print('SAVING MAKSKS STATS: ' + path_to_masks_statistics)
    print(amount_classes)
    pack = [absolute_time, amount_people, amount_classes['with_mask'], amount_classes['no_mask']]
    values_to_str = list(map(str, pack))

    with open(path_to_masks_statistics, 'a') as f:
        row = "{},{},{},{}\n".format(*values_to_str)
        f.write(row)

    print('SAVING DIST STATS: ' + path_to_contraventions_statistics)
    pack = [absolute_time, amount_people, contraventions]
    values_to_str = list(map(str, pack))

    with open(path_to_contraventions_statistics, 'a') as f:
        row = "{},{},{}\n".format(*values_to_str)
        f.write(row)


def null_data(**kwargs):
    """
    To nullify all counters when statistics has saved

    Args:
        init ([bool]): [true if u haven't counters and it's need to initialize them]

    Returns:
        [tuple]: 
            amount_people = 0, 
            amount_classes = {'label1': 0, ..., 'labelx': 0}, 
            iteration_start_time = 0
    """
    amount_people = 0
    amount_classes = {
        'with_mask': 0,
        'no_mask': 0
    }
    violation_count = 0
    iteration_start_time = time.time()

    print(bc.OKBLUE + 'Counters wass nulled' + bc.ENDC)

    return amount_people, amount_classes, violation_count, iteration_start_time


def update_line_graphics(**kwargs):
    """
    update linear plots in special directory

    Kwargs:
        data ([pandas.DataFrame]): [pandas table with all needed columns]
        path_to_masks_statistics ([str]): [path to statistic file]
        path_to_graphics_dir ([str]): [path to save graphics]
    """
    data = kwargs.get('data', False)

    if not data:
        path_to_statistic_file = kwargs.get('path_to_masks_statistics')
        data = pd.read_csv(path_to_statistic_file, engine='python')

    path_line_graphics = '{}/graph-1-1.html'.format(kwargs['path_to_graphics_dir'])

    fig = go.Figure()
    # saving plot for 'no_mask' and 'with_mask'
    fig.add_trace(go.Scatter(
        x=data.time, y=data.with_mask,
        text=data.with_mask, line_color='lightgreen',
        mode='lines+markers', name='with mask'
    ))
    fig.add_trace(go.Scatter(
        x=data.time, y=data.no_mask,
        text=data.no_mask, line_color='crimson',
        mode='lines+markers', name='no mask'
    ))
    # plot design update
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=False, width=700, height=150,
        legend=dict(x=-.1, y=1))

    fig.write_html(path_line_graphics)
    print(bc.OKBLUE + 'Graphics was saved to : {}'.format(path_line_graphics) + bc.ENDC)


def update_piechart(**kwargs):
    """
    update piechert plots in special directory

    Kwargs:
        labels ([List[str]]): [classification labels]
        path_to_masks_statistics ([str]): [path to statistic file]
        path_to_graphics_dir ([str]): [path to save graphics]
        data ([pandas.DataFrame]): [pandas table with all needed columns]
    """
    labels = kwargs.get('labels', ['С масками', 'Без масок'])
    data = kwargs.get('data', False)

    if not data:
        path_to_statistic_file = kwargs.get('path_to_masks_statistics')
        data = pd.read_csv(path_to_statistic_file, engine='python')

        print(bc.OKBLUE + 'UPDATE BARCHART: Data loading from {}'.format(path_to_statistic_file) + bc.ENDC)

    path_piechart = '{}/graph-1-2.html'.format(kwargs['path_to_graphics_dir'])
    values = [data['with_mask'].sum(), data['no_mask'].sum()]

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=.3)])

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=False, width=328, height=150,
        legend=dict(x=-.1, y=1.2)
    )
    fig.update_traces(
        marker=dict(colors=['lightgreen', 'crimson']))

    fig.write_html(path_piechart)
    print(bc.OKBLUE + 'Graphics was saved to : {}'.format(path_piechart) + bc.ENDC)


def update_barchart(**kwargs):
    """
    Update barchart plots in specified directory

    Kwargs:
        data ([pandas.DataFrame]): dataframe should consist columns, named: 
                                    'hours', 'all_people', 'contraventions'
        path_to_statistic_dir ([str]): path to statistic directory
        path_to_graphics_dir ([str]): directory path to save graphics
    """
    data = kwargs.get('data', False)

    if not data:
        path_to_contraventions_statistics = kwargs.get('path_to_contraventions_statistics')
        data = pd.read_csv(path_to_contraventions_statistics, engine='python')

        print(bc.OKBLUE + 'UPDATE BARCHART: Data loading from {}'.format(path_to_contraventions_statistics) + bc.ENDC)

    path_barchart = '{}/graph-1-3.html'.format(kwargs['path_to_graphics_dir'])

    data['hours'] = data['hours'].str[0:13]
    data = data.groupby('hours').sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data.hours,
        y=data.all_people,
        name='Amount people',
        marker_color='rgb(26, 118, 255)'
    ))
    fig.add_trace(go.Bar(
        x=data.hours,
        y=data.disturbance,
        name='Disturbances',
        marker_color='crimson'
    ))

    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        margin=dict(l=0, r=0, t=0, b=0),
        width=1100, height=300,
        xaxis_tickfont_size=14,
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.40,
        bargroupgap=0.1
    )

    fig.write_html(path_barchart)
    print(bc.OKBLUE + 'Graphics was saved to : {}'.format(path_barchart) + bc.ENDC)


def interval_timer(iteration_start_time, interval):
    time_diff = time.time() - iteration_start_time

    if time_diff > interval:
        print(bc.OKBLUE + 'New interval was detected' + bc.ENDC)

    return True if time_diff > interval else False


def concretize_statistics_files(**kwargs):
    """
    There is an function for defining path, that wasn't declare,
    but can be reached from already declared kwargs

    Returns:
        [ Dict[str, Any] ]: kwargs
    """
    path_to_statistics_dir = kwargs.get('path_to_statistics_dir', 'statistics/data')
    path_to_screenshots_dir = kwargs.get('path_to_screenshots_dir', 'statistics/screenshots')

    path_to_masks_statistics = '{}/masks_statistics_00.csv'.format(path_to_statistics_dir)
    path_to_contraventions_statistics = '{}/contraventions_statistics_00.csv'.format(path_to_statistics_dir)

    path_to_masks_screenshots = '{}/masks/'.format(path_to_screenshots_dir)
    path_to_contraventions_screenshots = '{}/distance/'.format(path_to_screenshots_dir)

    path_to_graphics_dir = kwargs.get('path_to_graphics_dir', 'graphics')

    kwargs['path_to_statistics_dir'] = path_to_statistics_dir
    kwargs['path_to_screenshots_dir'] = path_to_screenshots_dir
    kwargs['path_to_masks_statistics'] = path_to_masks_statistics
    kwargs['path_to_contraventions_statistics'] = path_to_contraventions_statistics
    kwargs['path_to_masks_screenshots'] = path_to_masks_screenshots
    kwargs['path_to_contraventions_screenshots'] = path_to_contraventions_screenshots
    kwargs['path_to_graphics_dir'] = path_to_graphics_dir

    return kwargs


def evaluate_statistics(**kwargs):
    """
    Main function for statistics evaluating

    **kwargs:
        amount_people: ([ int ]): amount of people, recognized at the video for the entire iteration
        amount_classes: ([ Dict[str, int] ]): dictionary containing the number of people 'with_mask' or 'no_mask'
        contraventions: ([ int ]): amount of contraventions, recognized at the video for the entire iteration

        absolute_time: ([ float ]): system time for logging
        iteration_start_time: ([ float ]): system time at starting concrete iteration
        interval: ([ int ]): time in seconds, defining iteration duration

        frame: ([ np.ndarray ]): frame with distance disturbance or None
        mask_frame: ([ np.ndarray ]): frame with mask disturbance or None

        path_to_statistic_dir: ([ str ]): path to file where masks statistics would be saved // 
                default value = {'statistics/data/mask_stats.csv'}
        path_to_graphics_dir: ([ str ]): path to file where statistical graphics would be saved // 
                default_value = {'graphics'} 
        path_to_screenshot_dir: ([ str ]): path to file where screenshots with contraventions would be saved // 
                default_value = {'statistics/screenshots/masks'} 

    
    Returns:
        counters ([ tuple ]): amount_people, amount_classes, violation_count, iteration_start_time
    """

    print(bc.OKBLUE + 'STATISTICS EVALUATION ENABLED' + bc.ENDC)

    transform_data = lambda t: time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(t))
    kwargs['absolute_time'] = kwargs.get('absolute_time', transform_data(time.time()))
Ы
    amount_people = kwargs.get('amount_people', 0)
    kwargs['amount_classes'] = kwargs.get('amount_classes', {'with_mask': 0, 'no_mask': 0})
    iteration_start_time = kwargs.get('iteration_start_time', 0)

    counters = amount_people, kwargs['amount_classes'], iteration_start_time
    kwargs = concretize_statistics_files(**kwargs)

    print(bc.OKBLUE + 'Called statisitcs writer to file and graphics updating' + bc.ENDC)

    write_data_to_file(**kwargs)

    update_line_graphics(**kwargs)
    update_piechart(**kwargs)
    update_barchart(**kwargs)

    # print(type(kwargs['frame']))

    if isinstance(kwargs['frame'], np.ndarray):
        print(bc.WARNING + 'Called SCREENSAVER FOR DISTANCE DIST' + bc.ENDC)
        save_screenshot(frame=kwargs['frame'], mode='d')

    if isinstance(kwargs['mask_frame'], np.ndarray):
        print(bc.WARNING + 'Called SCREENSAVER FOR MASK DIST' + bc.ENDC)
        save_screenshot(frame=kwargs['mask_frame'], mode='m')

    counters = null_data()

    print(bc.OKBLUE + 'Statistics sucsessfully writed to file: {} and graphics \
                        was updated'.format(kwargs['path_to_statistics_dir']) + bc.ENDC)

    return counters
