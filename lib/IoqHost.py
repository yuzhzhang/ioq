from __future__ import division

import sys, os
import time
import threading
import pickle
import datetime
import ConfigParser
import numpy as np

from get_logger import get_logger
from scheduler import Scheduler
from ptimer import PeriodicTimer

from diagiter2 import diagiter

from IceFactor import IceFactor as IoqFactor

logger = get_logger('IoqHost')

conn = None

_today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
_time_0925 = _today + datetime.timedelta(hours=9, minutes=25)
_time_0930 = _today + datetime.timedelta(hours=9, minutes=30)

class IoqHost(threading.Thread):
    def __init__(self):
        super(IoqHost, self).__init__()
        self.deamon = True
        self._stop = threading.Event()
        
        # Init
        self.suspended = False

        # Load Config
        config = ConfigParser.ConfigParser()
        config.read('../cfg/ioq.cfg')

        symbol_file = config.get('TRADE', 'SYMBOL_FILE')
        self.symbol_list = open(symbol_file).read().splitlines()
        self.symbol_dict = {sym:sym_id for sym_id, sym in enumerate(self.symbol_list)}
        self.symbol_price = {sym:0 for sym in self.symbol_list}
        self.symbol_bid = {sym:0 for sym in self.symbol_list}
        self.symbol_ask = {sym:0 for sym in self.symbol_list}
        self.symbol_ema_spread = {sym:0 for sym in self.symbol_list}
        self.N = len(self.symbol_list)

        self.pos = {sym:0 for sym in self.symbol_list}
        self.target_pos = {sym:0 for sym in self.symbol_list}
        self.final_target_pos = {sym:0 for sym in self.symbol_list}
        self.moo_target_pos = {sym:0 for sym in self.symbol_list}
        self.moo_pos = {sym:0 for sym in self.symbol_list}
        self.cash = {sym:0.0 for sym in self.symbol_list}
        self.bp = {sym:0.0 for sym in self.symbol_list}
        self.total_bp = 0.0
        self.volume = {sym: 0 for sym in self.symbol_list}

        self.ref_0928 = {sym:0 for sym in self.symbol_list}

        self.max_global_notional = config.getfloat('TRADE', 'MAX_GLOBAL_NOTIONAL')

        self.ioq_factor = [IoqFactor(sym) for sym in self.symbol_list]
        
        self.default_beta = 1.0
        self.beta = {sym:self.default_beta for sym in self.symbol_list}
        beta_file = config.get('TRADE', 'BETA_FILE')
        if os.path.isfile(beta_file):
            for row in open(beta_file).read().splitlines():
                words = row.split(',')
                sym = words[0]
                if sym in self.beta:
                    self.beta[sym] = float(words[1])
            logger.info('Loaded beta file.')
        else:
            logger.warn('Cannot find beta file %s.' % beta_file)

        self.prev_close = {sym:0 for sym in self.symbol_list}
        prev_close_file = config.get('TRADE', 'PREV_CLOSE_FILE')
        if os.path.isfile(prev_close_file):
            with open(prev_close_file) as closef:
                for line in closef:
                    if line.startswith('#'):
                        continue
                    try:
                        words = line.strip().split(',')
                        _sym = words[0].strip()
                        _px = np.float(words[-1])
                        if (_sym in self.symbol_list) and not(np.isnan(_px)):
                            self.prev_close[_sym] = _px
                            self.symbol_price[_sym] = _px
                    except:
                        continue
            logger.info('Loaded prev close file.')
        else:
            logger.error('Cannot find prev close file %s.' % prev_close_file)

        position_file = config.get('TRADE', 'POSITION_FILE')
        if os.path.isfile(position_file):
            with open(position_file) as posf:
                for line in posf:
                    if line.startswith('#'):
                        continue
                    words = line.strip().split(',')
                    _sym = words[0].strip()
                    _pos = int(words[1])
                    if _sym in self.symbol_list:
                        self.pos[_sym] = _pos
                        self.moo_pos[_sym] = _pos
                        self.cash[_sym] -= _pos * self.prev_close[_sym]
            logger.info('Loaded position file.')
        else:
            logger.warn('Cannot find position file %s.' % position_file)

        self.loo_ref_file = config.get('TRADE', 'LOO_REF_FILE')

        alpha_pkl_file = config.get('PRED_MODEL', 'ALPHA_PKL_FILE')
        if os.path.isfile(alpha_pkl_file):
            with open(alpha_pkl_file, 'rb') as pklf:
                self.alpha = pickle.load(pklf)
            logger.info('Loaded alpha pkl file.')
        else:
            logger.error('Cannot find alpha pkl file %s.' % alpha_pkl_file)
        
        Omega_pkl_file = config.get('PRED_MODEL', 'OMEGA_PKL_FILE')
        if os.path.isfile(Omega_pkl_file):
            with open(Omega_pkl_file, 'rb') as pklf:
                self.Omega = pickle.load(pklf)
            logger.info('Loaded Omega pkl file.')
        else:
            logger.error('Cannot find Omega pkl file %s.' % Omega_pkl_file)
 
        self.lambd = config.getfloat('PRED_MODEL', 'LAMBDA')
        self.c1 = config.getfloat('PRED_MODEL', 'COST_LINEAR')
        self.c2 = config.getfloat('PRED_MODEL', 'COST_QUAD')
        self.lb = config.getfloat('PRED_MODEL', 'LB')
        self.ub = config.getfloat('PRED_MODEL', 'UB')

        today = datetime.datetime.today().strftime('%Y%m%d')
        self.start_time = datetime.datetime.strptime(today+' '+config.get('TRADE', 'START_TIME'), '%Y%m%d %H:%M:%S')
        self.moo_cutoff_time = datetime.datetime.strptime(today+' '+config.get('TRADE', 'MOO_CUTOFF_TIME'), '%Y%m%d %H:%M:%S')
        self.loo_cutoff_time = datetime.datetime.strptime(today+' '+config.get('TRADE', 'LOO_CUTOFF_TIME'), '%Y%m%d %H:%M:%S')
        self.send_moo_time = datetime.datetime.strptime(today+' '+config.get('TRADE', 'SEND_MOO_TIME'), '%Y%m%d %H:%M:%S')
        self.send_loo_time = datetime.datetime.strptime(today+' '+config.get('TRADE', 'SEND_LOO_TIME'), '%Y%m%d %H:%M:%S')
        self.vwap_start_time = datetime.datetime.strptime(today+' '+config.get('TRADE', 'VWAP_START_TIME'), '%Y%m%d %H:%M:%S')
        self.vwap_end_time = datetime.datetime.strptime(today+' '+config.get('TRADE', 'VWAP_END_TIME'), '%Y%m%d %H:%M:%S')
        self.end_time = datetime.datetime.strptime(today+' '+config.get('TRADE', 'END_TIME'), '%Y%m%d %H:%M:%S')
        
        logger.info('VWAP start time: %s.' % self.vwap_start_time.strftime('%Y%m%d %H:%M:%S.%f'))
        logger.info('VWAP end time  : %s.' % self.vwap_end_time.strftime('%Y%m%d %H:%M:%S.%f'))

        self._moo_sched = Scheduler(self.send_moo_time, self.send_moo)
        self._loo_sched = Scheduler(self.send_loo_time, self.send_loo)
        self._exec_timer = PeriodicTimer(1.0, 0.0, self.send_target)
        self._sync_timer = PeriodicTimer(5.0, 0.5, self.summary)

        self.has_sent_moo = {sym:False for sym in self.symbol_list}
        self.is_oo_filled = {sym:False for sym in self.symbol_list}
       
        self.first_imb = {sym:0 for sym in self.symbol_list}

        self.print_config()
        
        self.check_prev_close()

        return
    
    def start_exec_timer(self):
        self._exec_timer.start()
        logger.info('Started exec timer.')
        return

    def start_sync_timer(self):
        self._sync_timer.start()
        return
    
    def check_prev_close(self):
        missing = []
        for sym in self.symbol_list:
            if self.prev_close[sym] < 0.1:
                missing += [sym]
        logger.warn('Missing prev close: %d symbols.' % len(missing))

    def print_config(self):
        report = '-' * 26 + '\n'
        report += 'Configure\n'
        report += '-' * 26 + '\n'
        report += '%-15s %8gmm\n' % ('Global BP', self.max_global_notional/1e6)
        report += '%-15s %10g\n'  % ('Lambda', self.lambd)
        report += '%-15s %10g\n'  % ('Cost Linear', self.c1)
        report += '%-15s %10g\n'  % ('Cost Quadratic', self.c2)
        report += '%-15s %10g\n'  % ('Lower Bound', self.lb)
        report += '%-15s %10g\n'  % ('Upper Bound', self.ub)
        report += '-' * 26
        for line in report.splitlines():
            logger.info(line)
        return

    def set_auct_sched(self):
        #self.set_moo_sched()
        self.set_loo_sched()
        return

    def set_moo_sched(self):
        self._moo_sched.start()
        logger.info('MOO scheduled at %s' % self.send_moo_time.strftime('%Y%m%d %H:%M:%S.%f'))
        return

    def set_loo_sched(self):
        self._loo_sched.start()
        logger.info('LOO scheduled at %s' % self.send_loo_time.strftime('%Y%m%d %H:%M:%S.%f'))
        return

    def on_market(self, market):
        if self.is_stopped():
            return
        sym = market['sym']
        price = market['price']
        ask = market['ask']
        bid = market['bid']
        ema_spread = market['ema_spread']
        if (ask < 0.1) or (bid < 0.1):
            return
        if sym in self.symbol_list:
            self.symbol_price[sym] = price
            self.symbol_bid[sym] = bid
            self.symbol_ask[sym] = ask
            self.symbol_ema_spread[sym] = ema_spread
        return
        
    def on_imba(self, imba):
        if self.is_stopped():
            return
        curr_time = datetime.datetime.now()
        if curr_time < self.start_time:
            return
        
        sym = imba['sym']
        if sym in self.symbol_dict:
            ioq_factor = self.ioq_factor[self.symbol_dict[sym]]
            is_total_update = imba['total'] * imba['side'] == ioq_factor.total * ioq_factor.side
            is_pair_update = imba['pair'] == ioq_factor.pair
            ioq_factor.update(imba)
            logger.info("Imba Update - %s (%.4f,%.0f,%.0f,%.0f,%d)" % \
                    (sym, imba['ref'], imba['pair'], imba['total'], imba['mkt'], imba['side']))
            
            if is_total_update:
                if (self.first_imb[sym] == 0) and (curr_time >= self.start_time): 
                    self.first_imb[sym] = imba['total'] * imba['side']
                    logger.info('Record first imbalance at 09:25:00 (%s,%.0f)' % (sym, self.first_imb[sym]))

        return

    def on_fill(self, fill):
        if self.is_stopped():
            return
        sym = fill['sym']
        size = fill['size']
        price = fill['price']
        self.pos[sym] += size
        self.cash[sym] -= size*price
        self.volume[sym] += abs(size)
        return

    def on_filloo(self, fill):
        if self.is_stopped():
            return
        self.on_fill(fill)
        sym = fill['sym']
        size = fill['size']
        self.moo_pos[sym] += size
        self.is_oo_filled[sym] = True
        return

    def send_moo(self):
        global conn
        if not(conn):
            logger.error('Failed to send MOO. Broken pipe.')
            return
        self.evaluate()
        order_msg_all = ''
        for sym in self.symbol_list:
            order_size = self.moo_target_pos[sym] - self.pos[sym]
            order_msg = '[MOO,%s,%d]|' % (sym, order_size)
            order_msg_all += order_msg
        if order_msg_all:
            conn.send(order_msg_all + '\n')
            for order_msg in order_msg_all.split('|')[:-1]:
                sym = order_msg.lstrip('[').rstrip(']').split(',')[1]
                self.has_sent_moo[sym] = True        
                logger.info(order_msg)
        logger.info('MOO orders sent.')
        return

    def send_loo(self):
        global conn
        if not(conn):
            logger.error('Failed to send LOO. Broken pipe.')
            return
        self.evaluate()
        ref_0928 = {}
        if os.path.isfile(self.loo_ref_file):
            with open(self.loo_ref_file) as reff:
                for line in reff:
                    words = line.strip().split(',')
                    sym = words[0]
                    ref = float(words[1])
                    ref_0928[sym] = ref
        order_msg_all = ''
        loo_count = 0
        for sym in self.symbol_list:
            if sym in ref_0928:
                _ref = ref_0928[sym]
            else:
                _ref = self.symbol_price[sym]
            if sym in self.prev_close:
                _prev_close = self.prev_close[sym]
            else:
                _prev_close = np.nan
            order_size = self.moo_target_pos[sym] - self.pos[sym]
            if order_size == 0:
                continue
            if order_size > 0:
                loo_price = np.nanmax([_ref, _prev_close]) * 1.05
                if np.isnan(loo_price):
                    continue
            elif order_size < 0:
                loo_price = np.nanmin([_ref, _prev_close]) * 0.95
                if np.isnan(loo_price):
                    continue
            order_msg = '[LOO,%s,%d,%.2f]|' % (sym, order_size, loo_price)
            order_msg_all += order_msg
            loo_count += 1
        if order_msg_all:
            conn.send(order_msg_all + '\n')
            for order_msg in order_msg_all.split('|')[:-1]:
                sym = order_msg.lstrip('[').rstrip(']').split(',')[1]
                self.has_sent_moo[sym] = True        
                logger.info(order_msg)
        logger.info('LOO orders sent.')
        return

    def evaluate(self):

        curr_time = datetime.datetime.now()
        
        logger.info('Start evaluation ...')
        
        N = len(self.symbol_list) 
        v = np.zeros([N, 1])
        lb = self.lb * np.ones([N, 1])
        ub = self.ub * np.ones([N, 1])
        c1 = self.c1 * np.ones([N, 1])

        logger.info('Predicting...')
        yhat = np.zeros([N, 1])
        yscale = np.zeros([N, 1])
        for sym in self.symbol_list:
            sym_id = self.symbol_dict[sym]
            
            ref = self.ioq_factor[sym_id].ref
            if ref < 0.1:
                price = self.symbol_price[sym]
                ref = price
            if ref < 0.1:
                #logger.warn('No price found for %s.' % sym)
                self.target_pos[sym] = 0
                continue
            
            v[sym_id] = self.pos[sym] * ref / self.max_global_notional
            
            if self.prev_close[sym] < 0.1:
                self.target_pos[sym] = 0
                lb[sym_id] = v[sym_id]
                ub[sym_id] = v[sym_id]
                continue
            
            curr_imb = self.ioq_factor[sym_id].total * self.ioq_factor[sym_id].side
            curr_pair = self.ioq_factor[sym_id].pair
            imb_diff = curr_imb - self.first_imb[sym]
            overnight_ret = np.log(ref) - np.log(self.prev_close[sym])
            isvalid = (curr_pair >= 1000) & ((np.abs(curr_imb) >= 100) | (np.abs(imb_diff) >= 100)) & (np.abs(overnight_ret) < 0.05)
            if isvalid:
                x1 = np.maximum(-0.025, np.minimum(0.025, overnight_ret))
                x2 = 1.0 * imb_diff / (np.abs(imb_diff) + curr_pair)
                x3 = 1.0 * curr_imb / (np.abs(curr_imb) + curr_pair)
                #logger.info('Details: %s x1 %f x2 %f x3 %f' % (sym, x1, x2, x3))
                X = np.array([x1, x2, x3])
                if np.all(~np.isnan(X)):
                    X = X[None, :]
                    _yhat, _yscale = self.alpha.predict(X)
                    yhat[sym_id] = _yhat
                    yscale[sym_id] = _yscale
                    #logger.info('pred details: %s yhat %f yscale %f.' % (sym, _yhat, _yscale))
                pair_lmt = 0.1 * curr_pair * ref / self.max_global_notional
                lb[sym_id] = max(lb[sym_id], -pair_lmt)
                ub[sym_id] = min(ub[sym_id], pair_lmt)
            else:
                lb[sym_id] = v[sym_id]
                ub[sym_id] = v[sym_id]

        logger.info('Prediction finished.')
        
        valid_id = []
        df_D = self.Omega['D']
        df_V = self.Omega['V']
        D = np.zeros([N, 1])
        V = np.zeros([N, 1])
        for sym in self.symbol_list:
            sym_id = self.symbol_dict[sym]
            ref = self.ioq_factor[sym_id].ref
            if ref < 0.1:
                price = self.symbol_price[sym]
                ref = price
            if ref < 0.1:
                continue
            if (sym in df_D) and (sym in df_V) and not(np.isnan(yhat[sym_id])) and not(np.isnan(yscale[sym_id])):
                D[sym_id] = df_D.iloc[0][sym]
                V[sym_id] = df_V.iloc[0][sym]
                if D[sym_id] > 1e-8:
                    valid_id += [sym_id]
        
        D = D[valid_id]
        V = V[valid_id]
        yhat = yhat[valid_id]
        yscale = yscale[valid_id]
        v = v[valid_id]
        lb = lb[valid_id]
        ub = ub[valid_id]
        c1 = c1[valid_id]
        
        D = np.diag(D.flatten() ** 2 + np.median(D) ** 2)
        W = V.dot(V.T)

        D *= 1e4
        W *= 1e4

        logger.info('Optimizing...')
        u, exit_flag = diagiter(D, W, yhat, self.lambd, c1, self.c2, v, lb, ub)
        if exit_flag:
            logger.warn('Optimization stopped early due to iteration or runtime limit.')
        else:
            logger.info('Optimization finished.')

        for idx, sym_id in enumerate(valid_id):
            sym = self.symbol_list[sym_id]
            
            ref = self.ioq_factor[sym_id].ref
            if ref < 0.1:
                price = self.symbol_price[sym]
                ref = price
            if ref < 0.1:
                self.moo_target_pos[sym] = 0
                continue
            
            moo_target_pos = u[idx] * self.max_global_notional / ref
            moo_target_pos = np.sign(moo_target_pos) * np.floor(np.abs(moo_target_pos) / 1.0) * 1
            
            self.moo_target_pos[sym] = moo_target_pos
        
        logger.info('Evaluation finished.')

        return
    
    def disable(self, sym):
        if sym in self.symbol_list:
            sym_id = self.symbol_dict[sym]
            self._is_disabled[sym] = True
            logger.info('Disabled %s.' % sym)
        return

    def enable(self, sym):
        if sym in self.symbol_list:
            sym_id = self.symbol_dict[sym]
            self._is_disabled[sym] = False
            logger.info('Enabled %s.' % sym)
        return
    
    def send_target(self):
        global conn
        if not(conn):
            logger.error('Failed to send target. Broken pipe.')
            return
        curr_time = datetime.datetime.now()
        if curr_time < self.vwap_start_time:
            for sym in self.symbol_list:
                self.target_pos[sym]  = self.moo_pos[sym]
                self.final_target_pos[sym]  = self.moo_pos[sym]
            logger.info('VWAP not started. Quit sending target.')
            return
        if curr_time > self.vwap_end_time:
            for sym in self.symbol_list:
                self.target_pos[sym] = 0
                self.final_target_pos[sym] = 0
        else:
            vwap_dur = (self.vwap_end_time - self.vwap_start_time).total_seconds()
            decay_rate = np.exp(np.log(0.5) / (vwap_dur / 5.0))
            delta_t = (curr_time - self.vwap_start_time).total_seconds()
            vwap_ratio = 0.8 * np.power(decay_rate, delta_t)
            for sym in self.symbol_list:
                target_pos = vwap_ratio * self.moo_pos[sym]
                target_pos = np.sign(target_pos) * np.floor(np.abs(target_pos) / 1.0) * 1
                self.target_pos[sym] = target_pos
                self.final_target_pos[sym] = 0
        logger.info('Sending target.')
        target_msg_all = ''
        for sym in self.symbol_list:
            if self.is_oo_filled[sym] or (curr_time > self.vwap_start_time + datetime.timedelta(seconds=4)):
                target_msg = '[TGT,%s,%.0f,%.0f]|' % (sym, self.target_pos[sym], self.final_target_pos[sym])
                target_msg_all += target_msg
        if target_msg_all:
            conn.send(target_msg_all + '\n')
            for target_msg in target_msg_all.split('|')[:-1]:
                logger.info(target_msg)
        return

    def summary(self):

        if conn:
            conn.send('[MKT]\n')
            logger.info('[MKT]')

        report = '%s\n' % (datetime.datetime.now().strftime('%Y%m%d %H:%M:%S.%f'))
        report += '%6s %8s %8s %10s %12s %12s %12s %12s %12s %12s %10s\n' % ('Symbol', 'Position', 'Target', 'PnL', 'Gmv', 'GmvTarget', 'Lmv', 'Smv', 'Beta', 'BP', 'Volume')
        body = ''
        total_pnl = 0.0
        total_gmv = 0.0
        total_gmv_tgt = 0.0
        #total_moo_tgt = 0.0
        total_lmv = 0.0
        total_smv = 0.0
        total_beta = 0.0
        total_volume = 0
        for sym in sorted(self.symbol_list):
            sym_id = self.symbol_dict[sym]
            pos = self.pos[sym]
            target_pos = self.target_pos[sym]
            #moo_target_pos = self.moo_target_pos[sym]
            if (pos == 0) and (abs(target_pos) < 1) and (abs(self.cash[sym]) == 0):
                continue
            
            ref = self.symbol_price[sym]
            if ref < 0.1:
                ref = self.ioq_factor[sym_id].ref
            pnl = self.cash[sym] + ref*pos
            lmv = ref*pos if pos>0 else 0.0
            smv = ref*pos if pos<0 else 0.0
            gmv = lmv - smv
            gmv_tgt = ref*abs(target_pos)
            beta_exposure = self.beta[sym] * ref * pos
            self.bp[sym] = max(gmv, self.bp[sym])

            total_pnl += pnl
            total_gmv += gmv
            total_gmv_tgt += gmv_tgt
            total_lmv += lmv
            total_smv += smv
            total_beta += beta_exposure
            total_volume += self.volume[sym]

            body += '%-6s %8.0f %8.0f %10.0f %12.0f %12.0f %12.0f %12.0f %12.0f %12.0f %10.0f\n' % (sym, pos, target_pos, pnl, gmv, gmv_tgt, lmv, smv, beta_exposure, self.bp[sym], self.volume[sym])
        
        self.total_bp = max(total_gmv, self.total_bp)

        report += '%-6s %8s %8s %10.0f %12.0f %12.0f %12.0f %12.0f %12.0f % 12.0f %10.0f\n' % ('*', '*', '*', total_pnl, total_gmv, total_gmv_tgt, total_lmv, total_smv, total_beta, self.total_bp, total_volume)
        report += body

        with open('../log/ioq_portfolio.snp', 'w') as snapfile:
            snapfile.write(report)
        with open('../log/ioq_portfolio.log', 'a') as logfile:
            logfile.write(report + '\n')
        
        return

    def is_stopped(self):
        return self._stop.isSet()

    def stop(self):
        self._stop.set()
        logger.info('Stopped.')
        return
    

