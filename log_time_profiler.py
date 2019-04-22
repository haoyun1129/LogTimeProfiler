#!/usr/bin/env python3
import fileinput
import argparse
import sys
import subprocess
from threading import Timer

import matplotlib.pyplot as plt
import numpy as np
import configparser
from datetime import datetime

if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more version is required.")


class LogTimeProfiler:
    action_timer: Timer
    t_request: int
    t_response: int

    def __init__(self):
        self.UPDATE_ON_FLY = True
        self.PIE_THRESHOLD_1 = 10
        self.PIE_THRESHOLD_2 = 15
        self.PIE_THRESHOLD_3 = 20
        self.ENCODING = "iso-8859-1"
        self.TITLE = 'Untitled'
        self.XLIM_LEFT = None
        self.XLIM_RIGHT = None
        self.t_request = 0
        self.t_response = 0
        self.t_current = 0
        self.t_last = 0
        self.t_session_start = 0
        self.MEASURE_START = ''
        self.MEASURE_END = ''
        self.PRINT_PATTERN = []
        self.verbose = False
        self.range_max = None
        self.range_min = None
        self.thresholds = []
        self.colors = []
        self.UPDATE_ON_THE_FLY = False
        self.command = None
        self.command_delay = 0
        self.command_count = -1
        self.action_timer = None
        self.measure_started = False
        self.line = ''
        self.test_count = 0
        self.log_output = None
        self.output_file = None

    @staticmethod
    def parse_time(time_string):
        time_string = time_string[:18]  # len("01-09 12:23:36.123") = 18
        dt_obj = datetime.strptime(time_string, '%m-%d %H:%M:%S.%f')
        return dt_obj.timestamp()

    def show_plot(self, hist):
        plt.clf()
        mu = np.mean(hist)
        sigma = np.std(hist)
        plt.suptitle(self.TITLE + r' - {} times, $\mu={:.2f}$, $\sigma={:.2f}$'.format(len(hist), mu, sigma))

        axes = plt.subplot(3, 1, 1)
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Times')
        n, bins, patches = plt.hist(hist)
        # y = stats.norm.pdf(bins, mu, sigma)
        # plt.plot(bins, y, '--', label='norm pdf')

        # axes.set_xlim(self.XLIM_LEFT, self.XLIM_RIGHT)

        axes = plt.subplot(3, 1, 2)
        plt.xlabel('Iteration')
        plt.ylabel('Response Time (seconds)')

        plt.scatter(range(len(hist)), hist)

        axes = plt.subplot(3, 1, 3)
        pie_dist = []
        labels = []
        for i in range(len(self.thresholds) - 1):
            count = [self.thresholds[i] < v <= self.thresholds[i + 1] for v in hist].count(True)
            if count > 0:
                pie_dist.append([self.thresholds[i] < v <= self.thresholds[i + 1] for v in hist].count(True))
                labels.append("{}~{}".format(self.thresholds[i], self.thresholds[i + 1]))

        plt.pie(pie_dist, autopct='%1.1f%%',
                labels=labels,
                colors=self.colors)
        plt.draw()
        plt.pause(0.001)

    def main(self):
        hist = []
        current = 0

        self.read_config_file()

        parser = argparse.ArgumentParser()

        parser.add_argument('--title', help='set the plot title')
        parser.add_argument('--encoding', help='E.g. iso-8895-1, utf-8')
        parser.add_argument('--fly', help='Update the plot on the fly', action='store_true')
        parser.add_argument('--verbose', help='Output verbosely', action='store_true')
        parser.add_argument('--run-command', help='Output verbosely', action='store_true')
        parser.add_argument('--range-max', help='set the max value to filter out the abnormal', type=int)
        parser.add_argument('--range-min', help='set the min value to filter out the abnormal', type=int)
        parser.add_argument('--xliml', help='The left xlim in data coordinates', type=float)
        parser.add_argument('--xlimr', help='The right xlim in data coordinates', type=float)
        parser.add_argument('--config', help='Config INI file')
        parser.add_argument('--output', help='Record log')
        parser.add_argument('file', metavar='FILE', help='files to read, if empty, stdin is used')
        args = parser.parse_args()

        if args.encoding:
            self.ENCODING = args.encoding
        if args.title:
            self.TITLE = args.title
        if args.range_max:
            self.range_max = args.range_max
        if args.range_min:
            self.range_min = args.range_min
        if args.xliml:
            self.XLIM_LEFT = args.xliml
        if args.xlimr:
            self.XLIM_RIGHT = args.xlimr
        if not args.run_command:
            self.command = None
        if args.output:
            self.log_output = args.output

        self.verbose = args.verbose
        self.UPDATE_ON_THE_FLY = args.fly
        if self.log_output:
            self.output_file = open(self.log_output, 'w')

        self.run_command()
        for line in fileinput.input(args.file, openhook=fileinput.hook_encoded(self.ENCODING)):
            line = line.strip()
            line_no = fileinput.lineno()
            self.line = line
            print_log = False
            print_current = False
            print_end = False

            for pattern in self.PRINT_PATTERN:
                if pattern in line:
                    self.t_current = LogTimeProfiler.parse_time(line)
                    print_log = True

            if not self.measure_started and self.MEASURE_START in line:
                self.measure_started = True
                print('measure_started =', self.measure_started)
                self.t_current = LogTimeProfiler.parse_time(line)
                self.t_request = self.t_current
                self.t_session_start = self.t_current
                print_log = True
            elif self.measure_started and self.MEASURE_END in line:
                self.measure_started = False
                print('measure_started =', self.measure_started)
                self.t_current = LogTimeProfiler.parse_time(line)
                self.t_response = self.t_current
                print_log = True
                current = self.t_response - self.t_request
                print(self.command, self.command_delay)
                if self.command and self.command_delay:
                    if self.action_timer:
                        self.action_timer.cancel()
                    self.action_timer = Timer(self.command_delay / 1000.0, self.run_command, [True])
                    self.action_timer.start()

                if current < 0:
                    self.print_log(line_no, line)
                    continue
                if self.range_min and current < self.range_min:
                    self.print_log(line_no, line)
                    continue
                if self.range_max and current > self.range_max:
                    self.print_log(line_no, line)
                    continue
                print_current = True
                print_end = True
                hist.append(int(current))
                self.test_count += 1

                if self.UPDATE_ON_THE_FLY:
                    self.show_plot(hist)

            if self.verbose and print_log:
                self.print_log(line_no, self.t_current - self.t_request, line)
            if print_current:
                self.print_log(current)
            if print_end:
                self.print_log('-' * 80)
            if self.command and self.command_count == self.test_count:
                break

        self.print_log('=' * 10, self.TITLE, 'Summary', '=' * 10)
        self.print_log('Result Count: {}'.format(len(hist)))
        self.print_log(
            'Benchmark: max = {}, min = {}, mean = {:.2f}, std = {:.2f}, mode = {:.2f}'.format(max(hist), min(hist),
                                                                                               np.mean(hist),
                                                                                               np.std(hist),
                                                                                               np.median(hist)))
        self.print_log(hist)
        self.show_plot(hist)
        plt.show()

    def read_config_file(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        if 'TRIGGER' in config:
            trigger = config['TRIGGER']
            if 'COMMAND' in trigger:
                self.command = trigger['COMMAND']
            if 'DELAY' in trigger:
                self.command_delay = int(trigger['DELAY'])
            if 'COUNT' in trigger:
                self.command_count = int(trigger['COUNT'])
        self.MEASURE_START = config['MEASURE']['START']
        self.MEASURE_END = config['MEASURE']['END']

        if 'PRINT' in config:
            for k, v in config['PRINT'].items():
                self.PRINT_PATTERN.append(v)
            if 'LOG_OUTPUT' in config['PRINT']:
                self.log_output = config['PRINT']['LOG_OUTPUT']
        for k, v in config['CHART'].items():
            if k.startswith('threshold'):
                self.thresholds.append(int(v))
        self.thresholds.sort()
        for k, v in config['CHART'].items():
            if k.startswith('color'):
                self.colors.append('#' + v)

    def run_command(self, start_measure=False):
        if self.command:
            print('run_command: ', '-' * 10, self.command, '-' * 10)
            subprocess.call(self.command, shell=True)
            if start_measure:
                self.measure_started = True
                print('measure_started =', self.measure_started)
                self.t_current = LogTimeProfiler.parse_time(self.line)
                self.t_request = self.t_current
                self.t_session_start = self.t_current

    def print_log(self, *objects):
        print(*objects)
        if self.output_file:
            self.output_file.write(' '.join([str(e) for e in objects]) + '\n')


if __name__ == '__main__':
    tp = LogTimeProfiler()
    tp.main()
