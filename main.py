import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from pylab import *

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
from template_mhe import template_mhe

import do_mpc


class MPC(QMainWindow):
    def __init__(self):
        super(MPC, self).__init__()
        
        # load ui file
        self.ui = uic.loadUi('ui.ui', self)

        self.ui.pushButton_apply.clicked.connect(self.apply)
        self.ui.pushButton_random_state.clicked.connect(self.random_state)
        self.ui.pushButton_next.clicked.connect(self.update)
        self.ui.pushButton_reset.clicked.connect(self.reset)
        self.ui.pushButton_play.clicked.connect(self.play)
        self.ui.pushButton_pause.clicked.connect(self.pause)
        self.ui.radioButton_random.toggled.connect(self.setPointRandom)

        self.ui.pushButton_pause.setEnabled(False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_graph)

        self.tick = 0

        self.isPlay = False

        # initial function
        self.setup()
        

    def setup(self):
        self.data_config = {
            "horizon" : 20,
            "sampling" : 0.1,
            "constrain_motor_lower" : -2,
            "constrain_motor_upper" : 2,
            "constrain_disk_lower" : -2,
            "constrain_disk_upper" : 2,
            "c1" : 2.697,
            "c2" : 2.66,
            "c3" : 3.05,
            "c4" : 2.86,
            "d1" : 6.78,
            "d2" : 8.01,
            "d3" : 8.82,
            "i1" : 2.25,
            "i2" : 2.25,
            "i3" : 2.25,
            "x0" : [0,0,0,0,0,0,0,0],
            "set-point" : "null"
        }
        self.ui_config = {
            "horizon" : self.ui.Input_horizon,
            "sampling" : self.ui.Input_sampling,
            "constrain_motor_lower" : self.ui.Input_constrain_motor_lower,
            "constrain_motor_upper" : self.ui.Input_constrain_motor_upper,
            "constrain_disk_lower" : self.ui.Input_constrain_disk_lower,
            "constrain_disk_upper" : self.ui.Input_constrain_disk_upper,
            "c1" : self.ui.Input_c1,
            "c2" : self.ui.Input_c2,
            "c3" : self.ui.Input_c3,
            "c4" : self.ui.Input_c4,
            "d1" : self.ui.Input_d1,
            "d2" : self.ui.Input_d2,
            "d3" : self.ui.Input_d3,
            "i1" : self.ui.Input_i1,
            "i2" : self.ui.Input_i2,
            "i3" : self.ui.Input_i3,
            "x0" : [self.ui.input_x0_1,self.ui.input_x0_2,self.ui.input_x0_3,self.ui.input_x0_4,self.ui.input_x0_5,self.ui.input_x0_6,self.ui.input_x0_7,self.ui.input_x0_8],
            "set-point" : self.ui.input_setpoint,
        }

        self.random_state()

    def setPointRandom(self):
        if self.ui.radioButton_random.isChecked():
            self.ui.input_setpoint.setEnabled(False)
            self.ui.input_setpoint.setText("null")
        else:
            self.ui.input_setpoint.setEnabled(True)
            self.ui.input_setpoint.setText("0.0")

    def random_state(self):
        self.data_config["x0"] = list((np.random.rand(8)-0.5).round(decimals=2))
        self.fill_config()
    
    def fill_config(self):
        for (key) in self.data_config:
            if (type(self.data_config[key]) == list):
                for i in range(len(self.data_config[key])):
                    self.ui_config[key][i].setText(str(self.data_config[key][i]))
            else:
                self.ui_config[key].setText(str(self.data_config[key]))

    def get_data_config(self):
        for (key) in self.data_config:
            if (type(self.data_config[key]) == list):
                
                for i in range(len(self.data_config[key])):
                    self.data_config[key][i] = float(self.ui_config[key][i].text())
            else:
                try:
                    self.data_config[key] = float(self.ui_config[key].text())
                except:
                    self.data_config[key] = "null"
    def create_figure(self):

        for i in reversed(range(self.ui.v_layout_line.count())): 
            self.ui.v_layout_line.itemAt(i).widget().setParent(None)
            
        for i in reversed(range(self.ui.v_layout_pie.count())): 
            self.ui.v_layout_pie.itemAt(i).widget().setParent(None)
            
        self.graph = []
        self.canvas = []

        for i in range(6):
            if i == 0 :
                fig, ax = plt.subplots(3,1, sharex=True)
            else:
                fig, ax = plt.subplots()
            self.graph.append([fig, ax])
            self.canvas.append(FigureCanvas(fig))
            if i == 0 :
                self.ui.v_layout_line.addWidget(self.canvas[-1])
            else:
                self.create_circle_data(i, ax, 0)
                ax.set_axis_off()
                ax.set_aspect('equal', 'box')
                self.ui.v_layout_pie.addWidget(self.canvas[-1])
    
    def create_circle_data(self, index , ax , radian):
        r = 5
        color =  "-bo"
        circle_color = "green"
        if index == 1 or index == 5 :
            circle_color = "grey"
        if index == 3 :
            color =  "-ro"
        an = np.linspace(0, 2 * np.pi, 100)
        ax.plot(r * np.cos(an), r * np.sin(an), circle_color, [0, r * np.cos(radian)], [0, r * np.sin(radian)], color)
        ax.set_axis_off()
        ax.set_aspect('equal', 'box')

    def make_step(self):
        self.u0 = self.mpc.make_step(self.x0)
        y_next = self.simulator.make_step(self.u0, v0=1e-2*np.random.randn(self.model.n_v,1))
        self.x0 = self.mhe.make_step(y_next)

        self.mpc_plot.reset_axes()
        self.mhe_plot.reset_axes()
        self.sim_plot.reset_axes()

        self.mpc_plot.plot_results()
        self.mpc_plot.plot_predictions()
        self.mhe_plot.plot_results()
        self.sim_plot.plot_results()

        self.graph[0][0].canvas.draw()

        self.graph[1][1].clear()
        self.create_circle_data(1, self.graph[1][1],self.x0[6][0])
        self.graph[1][0].canvas.draw()

        self.graph[5][1].clear()
        self.create_circle_data(5, self.graph[5][1],self.x0[7][0])
        self.graph[5][0].canvas.draw()

        self.graph[2][1].clear()
        self.create_circle_data(2, self.graph[2][1],self.x0[0][0])
        self.graph[2][0].canvas.draw()

        self.graph[3][1].clear()
        self.create_circle_data(3, self.graph[3][1],self.x0[1][0])
        self.graph[3][0].canvas.draw()

        self.graph[4][1].clear()
        self.create_circle_data(4, self.graph[4][1],self.x0[2][0])
        self.graph[4][0].canvas.draw()

    def update(self):
        self.make_step()

    def play(self):
        if self.isPlay:
            self.ui.pushButton_pause.setEnabled(True)
        else:
            self.timer.start(1000)
            self.isPlay = True
            self.ui.pushButton_play.setEnabled(False)
            self.ui.pushButton_next.setEnabled(False)
            self.ui.pushButton_reset.setEnabled(False)
            self.ui.pushButton_apply.setEnabled(False)
            self.ui.pushButton_pause.setEnabled(True)

    def play_graph(self):
        self.make_step()

    def pause(self):
        if self.isPlay:
            self.isPlay = False
            self.ui.pushButton_play.setEnabled(True)
            self.ui.pushButton_next.setEnabled(True)
            self.ui.pushButton_reset.setEnabled(True)
            self.ui.pushButton_apply.setEnabled(True)
            self.timer.stop()
            self.ui.pushButton_pause.setEnabled(False)
        else :
            self.ui.pushButton_pause.setEnabled(False)

    def apply(self):
        self.get_data_config()
        self.fill_config()
        self.create_figure()
        self.set_up_mpc()

    def reset(self):
        self.create_figure()
        self.set_up_mpc()

    def set_up_mpc(self):

        self.model = template_model(self.data_config)
        self.mpc = template_mpc(self.model, self.data_config)
        self.simulator = template_simulator(self.model, self.data_config)
        self.mhe = template_mhe(self.model, self.data_config)

        x0_true = np.array(self.data_config["x0"])
        self.x0 = np.zeros(self.model.n_x)

        self.mpc.x0 = self.x0
        self.simulator.x0 = x0_true
        self.mhe.x0 = self.x0
        self.mhe.p_est0 = 1e-4

        # Set initial guess for MHE/MPC based on initial state.
        self.mpc.set_initial_guess()
        self.mhe.set_initial_guess()

        color = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = self.graph[0]

        self.mpc_plot = do_mpc.graphics.Graphics(self.mpc.data)
        self.mhe_plot = do_mpc.graphics.Graphics(self.mhe.data)
        self.sim_plot = do_mpc.graphics.Graphics(self.simulator.data)

        ax[0].set_title('controlled position:')
        self.mpc_plot.add_line('_x', 'phi_2', ax[0])
        self.mpc_plot.add_line('_tvp', 'phi_2_set', ax[0], color=color[0], linestyle='--', alpha=0.5)

        ax[0].legend( self.mpc_plot.result_lines['_x', 'phi_2']+self.mpc_plot.result_lines['_tvp', 'phi_2_set']+self.mpc_plot.pred_lines['_x', 'phi_2'],
                ['Recorded', 'Setpoint', 'Predicted'], title='Disc 2')

        ax[1].set_title('uncontrolled position:')
        self.mpc_plot.add_line('_x', 'phi_1', ax[1])
        self.mpc_plot.add_line('_x', 'phi_3', ax[1])

        ax[1].legend(
            self.mpc_plot.result_lines['_x', 'phi_1']+self.mpc_plot.result_lines['_x', 'phi_3'],
            ['Disc 1', 'Disc 3']
        )

        ax[2].set_title('Inputs:')
        self.mpc_plot.add_line('_u', 'phi_m_set', ax[2])

        for mhe_line_i, sim_line_i in zip(self.mhe_plot.result_lines.full, self.sim_plot.result_lines.full):
            mhe_line_i.set_color(sim_line_i.get_color())
            sim_line_i.set_alpha(0.5)
            sim_line_i.set_linewidth(5)
        
        ax[0].set_ylabel('disc \n angle [rad]')
        ax[1].set_ylabel('disc \n angle [rad]')
        ax[2].set_ylabel('motor \n angle [rad]')

        for ax_i in ax:
            ax_i.axvline(1.0)

        fig.tight_layout()

        





if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    Dialog = MPC()
    Dialog.setWindowTitle("MPC Simulator")
    Dialog.showMaximized()
    sys.exit(app.exec_())
