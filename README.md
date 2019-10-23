# Antiferromagnetics for analogue neuromorphic computing
This directory contains simulations of antiferromagnetics for analogue neuromorphic computing. A European patent has be filled for this invention (EP19199322.9). The software and other relavent details to the inventoin will be realsed when the legal team has given me the ok.  For now below is an introduction to the topics. 

## Introduction to analogue neuromorphic computing
To the best of the author’s knowledge this is the first report investigating routes to implement neuromorphic computing with AFM counters. 

Digital computers have been extremely successful as they are Turing completeness machines, meaning that a sufficiently large one would be capable of performing any algorithm. Even though digital computers are Turing complete, this does not mean that the machine could solve all problems efficiently. To solve these difficult problems more effort has been focused on quantum and analogue computing which are more suitable to solve such problems. Quantum computers use quantum phenomena to compute, and analogue computers continuously solve a set of differential equations. By these definitions computers such as the D-Wave quantum computer are an example of both a quantum and analogue computer[1]. Here the focus is on analogue computers using classical physics, as they have the potential advantages of room temperature operation, portability and scalability. 

Historically, the challenge for analogue computers was to find general purpose algorithms where digital algorithms are insufficient. The most common use of analogue computers is to solve complex differential equations efficiently with some research still ongoing[2]. Classical analogue computing has recently undergone a renewal, as new paradigms such as memcomputing, neuromorphic computing and probabilistic computing of probabilistic bits (p-bits) have been proposed to solve problems such as factorization, pattern recognition and “stoquastic” Hamiltonians[3]–[6]. The outstanding issue in building these new analogue systems is the miniaturization of analogue components, which is something that is addressed in this work. Next we will focus on neuromorphic computing, which uses uniquely designed analogue, digital or mixed computers to simulate artificial neural networks (ANNs) with full or abstracted functionality of biological brains. 

Artificial neural networks are remarkably proficient at quantifying different tasks that a human would consider to be unquantifiable. These ANN algorithms are almost always implemented in digital hardware and have granted computers the ability to produce art, master GO, translate language, operate vehicles and even generate general artificial intelligence[7], [8]. There are now a large variety of dedicated digital neuromorphic processors such as TrueNorth by IBM, Loihi by Intel and SpiNNAker by the University of Manchester. It is an open research question whether the full potential of biological neural networks can be achieved with digital architecture, but this seems unlikely as the functionality of the biological neurons is time dependent, extremely parallel, uses random noise and can be described as continuous differential equations. For these specific reasons analogue computers appear to be a more appropriate architecture. 

As the conventional silicon building blocks for analogue circuits do not easily miniaturize, for a complete neuromorphic capable of learning, new materials systems are investigated. The most commonly suggested new materials for neuromorphic computers are oxide memristors which act as analogue resistive counters. For a recent and relevant review on memristors for analogue computing please refer to F. Caravelli et al. [9]. Interestingly, it appears that there are a group of AFM materials, such as CuMnAs and Mn2Au, which also act as electrically controllable counters which have improved endurance, fabrication reproducibility, fast switching rates, stray magnetic field sensitivity and thousands of deterministic memory states. As the AFM counters are particularly nonlinear, the question of whether the AFM memory can be used to learn is not trivial. In this report we determine whether the nonlinear characteristics of the AFM counter prevent it from being implemented in an analogue learning circuit. 

## Biological and artificial neurons
As the literature of neuromorphic computing heavily uses biological terms, it is important to understand the basics of neuron cell biology and how to apply them to digital and hardware based artificial neural networks. For the hardware of analogue neuromorphic computers, we will focus on all electrical networks.

### Neuron biology

There are many types of human biological neurons, however, all neurons have similar properties[10], [11]. Each neuron cell has dendrites receiving information from other neurons[10]. The dendrites collect either excitatory or inhibitory incoming signals, which either respectively increase or reduce the probability that the neuron will produce an output signal. The cell body (soma) contains the nucleus of the neuron and other structures that maintain homeostasis of the cell[10], [11]. Both the soma and the dendrites are used to process the signals from the presynaptic neurons. The membrane potential, the electrochemical potential difference between inside and outside the cell, stores the specific state of the neuron electrochemically. If the membrane potential reaches a set threshold, an output spike signal is sent through the axon to the target neurons and the membrane potential reduces[10]. A single neuron can have thousands of different axon terminals connecting the neuron to different postsynaptic neurons. The interface where the signal from the emitting neuron, known as the presynaptic neuron, is transmitted to the target neuron, known as the postsynaptic neuron, is called the synapse. The information is physically transferred between neurons using neuro-transmitters. The connection strength of the synapse is one of the features that determines the effect of the presynaptic neuron on the postsynaptic neuron[10]. The strength of the synapse depends upon the history of activity between both the presynaptic and postsynaptic neurons. The exact learning mechanisms of the neuron cells are still being researched and depend on the cell type. Known trainable features include membrane time constant, synapse strengths, thresholds and signal conduction delays[10].

 <p align="center">
    <img width="" height="" src="https://github.com/OE-FET/Neuromorphic_computing/blob/master/imgs/Neuron_bio.png">
</p>

Figure 1: Structure of a neuron cell. Image adapted from Yael Avissar [11].

### Artificial neurons
The artificial digital neuron is described mathematically as an activation function (f) which maps the weighted sum of the discrete input values (x_i) and bias value (b) to a discrete output value (y), i.e. <a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;f(b&plus;\sum_{i}^{n}w_{j,i}x_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;f(b&plus;\sum_{i}^{n}w_{j,i}x_{i})" title="y = f(b+\sum_{i}^{n}w_{j,i}x_{i})" /></a>. The synapse weights (<a href="https://www.codecogs.com/eqnedit.php?latex=w_{j,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{j,i}" title="w_{j,i}" /></a>) which connect the output of neuron i to the input of neuron j are analogous to the synapse strength between the presynaptic and postsynaptic neurons. The activation function used in applications depends upon the required computational speed and accuracy but an important aspect is maintaining nonlinearity. For example, single layer and deep neural networks with linear activation functions have identical model expressiveness which is not the case for nonlinear activation functions[12]. Common examples of activation functions include signum, sigmoid, heaviside step, rectal linear and hyperbolic tangent functions[8], [13]. 

An artificial analogue neuron can behave identically to a digital neuron during continuous time or to a more biologically plausible neuron model which simulates the membrane potential (v(t)), which is a time dependent quantity that is dependent on the history of the neuron output and input signals from other neurons and is directly related to the output of the neuron (Figure 2). 

The membrane potential can be mathematically identical to the digital neuron just in continuous time <a href="https://www.codecogs.com/eqnedit.php?latex=v_{j}(t)=b&plus;\sum_{i}^{n}&space;z_{j,i}x_{i}(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_{j}(t)=b&plus;\sum_{i}^{n}&space;z_{j,i}x_{i}(t)" title="v_{j}(t)=b+\sum_{i}^{n} z_{j,i}x_{i}(t)" /></a>, or can be more biologically plausible where the membrane potential is defined as a differential equation dependent on the history of the neuron <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\mathrm{d}&space;v_{j}(t)}{\mathrm{d}&space;t}&space;=&space;g(b,z_{j,1..n}(t'),x_{1..n}(t'),&space;y_{i}(t'))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\mathrm{d}&space;v_{j}(t)}{\mathrm{d}&space;t}&space;=&space;g(b,z_{j,1..n}(t'),x_{1..n}(t'),&space;y_{i}(t'))" title="\frac{\mathrm{d} v_{j}(t)}{\mathrm{d} t} = g(b,z_{j,1..n}(t'),x_{1..n}(t'), y_{i}(t'))" /></a>. The output from analogue neurons can be a continuous differentiable signal and/or a spike train which propagates to the input of connected neurons via a variable impedance element (https://latex.codecogs.com/gif.latex?z_%7Bi%2Cj%7D) which simulates the synapse weight. This work focuses on plausible routes to build an artificial variable impedance synapse using AFM and the exact details of the neuron circuitry are not discussed here. Possible neurons include an all CMOS neuron which consists of a leaky current integrator which outputs spike and zero resets the integrator when the integration reaches a set threshold, and Professor Datta Supriyo’s p-bit neuron which stochastically outputs spikes with a probability proportional to the sigmoid of the sum of the input signals from the presynaptic neurons[6], [14]. 

 <p align="center">
    <img width="" height="" src="https://github.com/OE-FET/Neuromorphic_computing/blob/master/imgs/Digital_analogue_Neuron.png">
</p>
Figure 2: Digital and analogue artificial neurons used in artificial neural networks.

### Artificial neural networks

Networks of neurons are generated by connecting the output of neurons to the input of other neurons or the same neuron. The properties of the networks are controlled by the number of neurons, activation function and network architecture[8]. Here the functionality and hardware implementations of network architectures is described.

#### Feedforward network 

Feedforward networks contain multiple hidden layers of neurons which connect the input layer of neurons to the output layer of neurons[15]. An example of a feed forward artificial neural network with one hidden layer is seen in Figure 3. Deep neural networks are networks with many hidden layers. Deep neural networks have become pervasive because each additional hidden layer enables the network to predict more complex and abstract functions. An example of a single layer of an analogue neural network using a form of variable resistor, memristors, for the artificial synapse is seen in Figure 3. Multiple analogue layers could be attached in series to simulate a deep neural network. To compare the computational effectiveness of the suggested analogue neural network architecture to a digital computer, we compare the number of computations required to simulate the current through each of the memristors in the analogue computer. If the computer contains 108 memristors (memristor area of 0.01μm2 on a 1mm2 substrate), the output of each neuron can be discretized into 1ns intervals, and if each instance of current through the memristor is considered a floating point operation, the analogue computer would efficiently perform 1017 flop digital computations which is on the same order as that of the world’s fastest super computer. 

  <p align="center">
    <img width="" height="" src="https://github.com/OE-FET/Neuromorphic_computing/blob/master/imgs/feed_forward.png">
</p>
Figure 3: Left) Feed forward multilayer digital neural network and right) single layer analogue neural network. Image adapted from C. Zamarreno-Ramos et al. [16].  

#### Convolution networks

Convolution neural networks perform a series of convolutions, nonlinear layers and pooling operations in order to generate a feature space which is an input to another neural network. The convolution layer convolves the inputs with specific kernels. The nonlinear layer applies an elementwise activation function and the pool layer performs a down sampling operation. Convolution neural networks have been particularly useful in the field of image processing and are inspired by convolution neural networks used in biological neurons in the visual cortex  of mammals[17]. The structure of convolutional neural networks can get rather complicated; for more information, please refer to Practical Convolutional Neural Networks by Sewak et al.[18]

#### Recurrent network 

Recurrent neural networks have neurons where the present state of the neuron is dependent on the previous states of the network. A possible route to achieve this is to connect the output layer of a feedforward network to an input of a neuron in the input layer. This enables recurrent neural networks to have temporal dynamics allowing them to use their internal memory to process a sequence of events, which makes them suitable for temporal tasks such as speech recognition or predicting the movement of objects across a series of images. Continuous recurrent neural networks are particularly suited for analogue ANNs as they operate in continuous time and direct electrical feedback can easily be achieved. 

#### Reservoir network

Reservoir neural networks contain a network of fixed interconnected neurons with either random or fixed synapse weights, called a reservoir, and an output layer to interpret the reservoir by sampling a subset of the reservoir neurons (Figure 4)[19]. Similar to recurrent networks, reservoir networks are suitable for temporal tasks. Reservoir networks are particularly interesting for analogue neuromorphic applications as learning does not occur by modification of the synapse weights in the reservoir and reservoir networks are mathematically equivalent to a collection of non-linear oscillators(Figure 4)[19]. These properties allow reservoir ANNs to be simulated in unintuitive systems such as electrical nanoscale spin-torque oscillators, optical m-z modulators, spatio-temporal photonic systems and physical buckets of water[19]–[22]. 

  <p align="center">
    <img width="" height="" src="https://github.com/OE-FET/Neuromorphic_computing/blob/master/imgs/Reservoir.png">
</p>
Figure 4: Left) Reservoir network using artificial neurons. Right) Schematic description of a nonlinear computing network. The N inertial masses (circles) arranged in a chain are coupled to neighbors by linear springs and to a substrate by a linear or non-linear spring, with damping. A harmonic forcing, with amplitude possibly modulated by coupling to the input signal u(t), is imposed on the masses. Image adapted from C. Coulombe et al. [23]. 

#### Bayesian neural networks 

Stochastic neural networks contain random variables built into the network[24]. This would include networks such as Markov Chain Neural Networks, Boltzmann machines or Bayesian neural networks. Here the focus is on Bayesian neural networks which have probabilistic synapse weights that are robust to over-fitting, can estimate uncertainty through a Monte Carlo simulation[25]. Furthermore, Yarin Gal et al. have demonstrated that Bayesian neural networks with Gaussian distributed probabilistic weights can be approximated by performing dropout on neurons, where dropout is a procedure used in ANNs to randomly exclude a neuron from a model[26]. Dropout could easily occur in an analogue network by continuously and randomly disconnecting a neuron using a stochastic tunnel junction as the random noise source[6]. 

#### Learning paradigms 

In order for the neural networks to learn, the synapse weights of the neural network need to be appropriately updated. As learning algorithms in common digital neural networks can be implemented using only multiplication, addition and subtraction operators, it is possible that some digital algorithms could be directly applied using analogue circuits. Here, the most common learning modalities in machine learning are discussed. In particular, time dependent spike algorithms which are not commonly implemented in digital applications as they are more suitable for analogue computers are investigated. 

Unsupervised learning finds commonalities in unlabeled data and updates the model based on the absence or presence of the commonalities in new data. A commonly proposed unsupervised learning algorithm for neuromorphic computers with biological  plausibility is spike time dependent plasticity (STDP)[27]–[33]. The basic STDP algorithm, seen in equation 1, determines that the change of the synapse weight <a href="https://www.codecogs.com/eqnedit.php?latex=w_{i,j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{i,j}" title="w_{i,j}" /></a> between presynaptic (i) and postsynaptic neurons (j) is dependent on the time difference <a href="https://www.codecogs.com/eqnedit.php?latex=(t_{i}-t_{j})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(t_{i}-t_{j})" title="(t_{i}-t_{j})" /></a> between the presynaptic and postsynaptic spikes and the learning curve (f)[30]. 

(1)					 <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_{i,j}&space;=f(t_{i}-t_{j})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_{i,j}&space;=f(t_{i}-t_{j})" title="\Delta w_{i,j} =f(t_{i}-t_{j})" /></a>

The common implementation of the STDP algorithm in neuromorphic chips occurs because memristors and specific presynaptic and postsynaptic waveforms are capable of implementing STDP. The basic implementation of STDP using memristors, using a similar architecture as that in Figure 7 3, can be applied only to a single layer of neurons or used to train readouts for reservoir networks. However, simple STDP memristor networks are not capable of performing XOR classification[31], [34]. In this work it is demonstrated that STDP learning is possible for AFM based synapses. More complex STDP algorithms will be reviewed in the next sections on supervised and reinforcement learning. 

Supervised learning maps input data to output data based on known input output data pairs. The process of learning occurs by modifying the set of synaptic weights between neurons and minimizing the cost function of the specific neural network. In certain spike algorithms such as SpikeProp and Remote Supervised Learning Method the timing of spike encodes the information and learning occurs through minimizing the timing error between an output spike and a desired target spike using multilayer STDP and/or backpropagation[35], [36]. The algorithms are difficult to implement in analogue computers as precise timing and/or knowledge of previous states of the network are required when a learning event is triggered.  

In a spike rate network (SRN) the output of a particular neuron is a train of Dirac-Delta pulses where the average spiking rate encodes the output value[37]. SRNs use rec-linear action functions and can be mathematically equivalent to digital artificial neural networks using rec-linear activation functions[8], [37]. In this report it is demonstrated that it is possible to build a multilayer spike rate network using AFM memory. The specific algorithm used was provided by Peter O’Connor et al. in their paper on Deep Spiking Networks[37].

### Summary

This section provided the fundamentals of how analogue neural networks work in nature, how to replicate these networks in hardware and how to implement different learning paradigms. Using this knowledge, the question of how to build an analogue computer, specifically the analogue synapses, is examined. 

## Analogue synapses

Here we review possible hardware technologies capable of acting as controllable analogue synapses for electrical neural networks and compare the properties of these analogue synapses to antiferromagnetic memories to determine whether an antiferromagnetic based synapse could provide any benefit. A simple resistor could act as a synapse but will be excluded, as a resistor has no learning capabilities and cannot be updated.  The electrical synapses reviewed include floating-gate MOSFETs similar to neuMOS, OLED inspired synapses, analogue electrically erasable programmable read-only memory (EEPROM) synapses as well as memristors. 	

The neuMOS floating-gate MOSFETs, referred to here as vMOS, were proposed in the mid-1990s as artificial neurons and recently they have been attracting attention as an artificial synapse in probabilistic computing[6], [38]. vMOS are variable voltage threshold MOSFETs where the voltage threshold is controlled by using n number of variable capacitors coupled to a MOSEFET with a floating gate. The voltage threshold scales linearly with the voltage applied to each of the capacitors. A single vMOS can be operated as a digitally controlled variable resistor. Similarly a complementary vMOS source-follower circuit acts as a voltage to voltage converter which can perform a weighted sum of all the output voltages of the presynaptic neurons. The weighted sum is the input to the postsynaptic neuron. The weight and synapse strength are controlled by the specific capacitors connecting the presynaptic neuron to complementary vMOS using a network of programmable switches. 

The OLED inspired synapse uses similar analogue transistor architectures, such as 2T1C, used in OLED transistor backplanes in order to control the current through the diode. However, instead of the transistor controlling the current through a diode like in an OLED display, the transistor controls the magnitude of the current from the presynaptic neuron to the postsynaptic neurons, therefore acting as a controllable synapse. To the best of the author's knowledge there is no reference to this specific architecture in the literature.

Interestingly, as OLED inspired and resistive vMOS  synapses are passive components, it is feasible to stack MOSFETs as the transistors are not actively consuming power. An outstanding problem about both vMOS and OLED inspired design is that learning via direct feedback from the presynaptic and postsynaptic neurons is not possible, but for a lot of applications learning is not necessary. This will be addressed later in this work. Learning requirements will be discussed in further detail later in the report but for practical applications they might not be necessary.  

Another plausible MOSFET based electric synapse uses electrically erasable programmable read-only memory (EEPROM), commonly used in flash memory, where a controlled amount of change is stored in the floating gate[39]. The resistance of the MOSFET is then dependent on the charge of the floating gate allowing for a large number of analogue states. Unlike the digital to analogue synapses mentioned above, controlling the waveforms on the device allows the synapse to perform algorithm learning. As electrons are added to the gate through hot-electron injection with silicon dioxide, over time the silicon oxide layer degrades. This degradation is the major limiting factor for the future of both digital and analogue based EEPROM memories. 

Memristors are the fourth fundamental passive circuit components where the conductance is dependent on the history of the current passing through the element[40]. In 2008, the first memristor devices were fabricated using TiO2 where the change in conductivity of the films was due to the migration of oxygen vacancies in the presence of an electric field over a minimum threshold[41]. Since 2007, more oxide, organic, ferroelectric and carbon nanotube based memristors have been reported[30], [31], [42]–[47]. Memristors are typically considered the natural synapse, as they are two terminal resistive devices that are capable of performing spike time dependent plasticity algorithms. Memristors perform STDP due to their inherent conductance change from constructive interference between the presynaptic and postsynaptic waveforms applied across the memristors, which results in a larger voltage differential across the memristor than either a single presynaptic or postsynaptic waveform[30]. It should be noted that memristor functionality can be completely replicated using CMOS circuits; however, fully CMOS memristors are active components that require large areas. 

In comparing the memory properties of analogue EEPROMs and the memrestive synapses capable of using feedback to learn, to the memory properties of antiferromagnetic memories, shown in Table 9, it appears there are certain aspects where AFM memories are superior. Antiferromagnetic memories have high endurance, high fabrication reproducibility, negligible sensitivity to stray fields, and reproducible memory states which suggest that the AFM memories could have unique advantages for specific applications. The details of AFM memories will be discussed in the following section. 

## Methods and Results

Temporarily removed.....

## Conclusion

Even though learning is feasible with AFM counters and other memristors, the specific future of the analogue neuromorphic computer is uncertain as there is not a clear consensus about direct applications where digital computers are insufficient. The long-term future of fully developed analogue computers is promising as analogue computers fundamentally use an appropriate basis function set to simulate biological and artificial neural networks. Therefore, it is conceivable that with the appropriate technology, analogue neuromorphic computers will fill a niche application alongside digital and quantum computers[49]. Here appropriate applications where neuromorphic computers could solve niche problems are suggested, such as improved continuous data acquisition and inference. 

The quality and quantity of data used in artificial intelligence significantly limit the predictive capabilities of models. However, in continuous monitoring of real-world assets, extremely large quantities of data are generated and only a subset of the data is stored due to limited storage. Neuromorphic computing could provide a low power approach to perform continuous analysis of real time data that would otherwise be ignored. 

To the best of the author’s knowledge almost all of the analogue neuromorphic computers in the literature are focused on learning, however, in most applications of neural networks, such as performing inference in image recognition in autonomous cars or texture generation in computer graphics, the state of the synapses is not modified when the network is preforming the inference. Google's first generation of tensor processing units took advantage of this aspect of neural networks and demonstrated large efficiency gains in performing inferences with only 8 bit precision[50]. Analogue computers should take a similar route, first demonstrating large computational gains through a low precision static network and then incorporating additional hardware to improve the precision and learn in situ. For example, a logical route to introduce learning would be to place an STDP network at the output of a static network pipeline in order to provide fast real time learning using convoluted features generated by a static network. 

The most novel insight of this work is that there are standard CMOS architectures likely capable of performing computationally demanding inferences.  More focus should be placed on demonstrating large computational gains of static networks which could have a profound impact on edge computing.



## References
[1]	J. King et al., “Quantum Annealing amid Local Ruggedness and Global Frustration,” 2017.

[2]	N. Guo et al., “Energy-Efficient Hybrid Analog/Digital Approximate Computation in Continuous Time,” IEEE J. Solid-State Circuits, vol. 51, no. 7, pp. 1514–1524, 2016.

[3]	F. L. Traversa, P. Cicotti, F. Sheldon, and M. Di Ventra, “Evidence of an exponential speed-up in the solution of hard optimization problems,” Complexity, vol. 2018, p. 13, 2017.

[4]	M. Di Ventra and F. L. Traversa, “Memcomputing: Leveraging memory and physics to compute efficiently,” J. Appl. Phys., vol. 123, no. 18, pp. 1–16, 2018.

[5]	K. Y. Camsari et al., “Scaled Quantum Circuits Emulated with Room Temperature p-Bits,” arXiv, vol. 47907, pp. 1–9, 2018.

[6]	K. Y. Camsari, B. M. Sutton, and S. Datta, “p-Bits for Probabilistic Spin Logic,” Phys. Rev. X, vol. 47907, pp. 1–10, 2017.

[7]	D. Hassabis, D. Kumaran, C. Summerfield, and M. Botvinick, “Neuroscience-Inspired Artificial Intelligence,” Neuron, vol. 95, no. 2, pp. 245–258, 2017.

[8]	M. A. Nielsen, Neural Networks and Deep Learning. Determination Press, 2015.

[9]	F. Caravelli and J. P. Carbajal, “Memristors for the curious outsiders,” Technologies, pp. 1–42, 2018.

[10]	I. Levitan and  ‎Leonard Kaczmarek, The Neuron: Cell and Molecular Biology, 3rd editio. Oxford University Press, 2001.

[11]	Yael Avissar, Biology. Houston, Texas: OpenStax College, 2013.

[12]	A. M. Saxe, “Deep linear neural networks: A theory of learning in the brain and mind,” Stanford University, 2015.

[13]	C. Hurtes, M. Boulou,  a. Mitonneau, and D. Bois, “Deep-level spectroscopy in high-resistivity materials,” Appl. Phys. Lett., vol. 32, no. 12, p. 821, 1978.

[14]	X. Wu, V. Saxena, and K. Zhu, “A CMOS Spiking Neuron for Dense Memristor- Synapse Connectivity for Brain-Inspired Computing,” in Neural Networks (IJCNN), 2015.

[15]	I. Goodfellow, B. Yoshua, and C. Aaron, Deep Learning. MIT Press, 2016.

[16]	C. Zamarreño-Ramos, L. A. Camuñas-Mesa, J. A. Perez-Carrasco, T. Masquelier, T. Serrano-Gotarredona, and B. Linares-Barranco, “On spike-timing-dependent-plasticity, memristive devices, and building a self-learning visual cortex,” Front. Neurosci., vol. 5, no. 26, pp. 1–22, 2011.

[17]	B. Y. D. H. Hubel and A. D. T. N. Wiesel, “Receptive fields, binocular interaction and functional architecture in the cat’s visual cortex,” J. Physiol., pp. 106–154, 1962.

[18]	M. Sewak, R. Karim, and P. Pujari, Pratical Convolutional Neural Networks, 1st ed. Mumbai: Packt Publishing, 2018.

[19]	B. Schrauwen, D. Verstraeten, and J. Van Campenhout, “An overview of reservoir computing: theory, applications and implementations,” in Proceedings of the 15th European Symposium on Artificial Neural Networks, 2007, pp. 471–82.

[20]	Y. Paquot et al., “Optoelectronic Reservoir Computing,” Sci. Rep., vol. 2, no. 1, p. 287, 2012.

[21]	Y. Shen, N. C. Harris, D. Englund, and M. Soljacic, “Deep learning with coherent nanophotonic circuits,” arXiv, 2016.

[22]	J. Bueno et al., “Reinforcement Learning in a large scale photonic Recurrent Neural Network,” arXiv, vol. 1, pp. 1–5, 2017.

[23]	J. C. Coulombe, M. C. A. York, and J. Sylvestre, “Computing with networks of nonlinear mechanical oscillators,” PLoS One, vol. 12, no. 6, pp. 1–13, 2017.

[24]	M. Awiszus and B. Rosenhahn, “Markov Chain Neural Networks,” arXiv, 2018.

[25]	Y. Gal, “Uncertainty in Deep Learning,” University of Cambridge, 2016.

[26]	Y. Gal and Z. Ghahramani, “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,” arXiv, vol. 48, 2015.

[27]	T. Serrano-Gotarredona, T. Masquelier, T. Prodromakis, G. Indiveri, and B. Linares-Barranco, “STDP and STDP variations with memristors for spiking neuromorphic learning systems,” Front. Neurosci., vol. 7, pp. 1–15, 2013.

[28]	C. Clopath and W. Gerstner, “Voltage and spike timing interact in STDP - a unified model,” Front. Synaptic Neurosci., 2010.

[29]	Y. Babacan and F. Kacar, “Memristor emulator with spike-timing-dependent-plasticity,” AEU - Int. J. Electron. Commun., vol. 73, pp. 16–22, 2017.

[30]	S. Boyn et al., “Learning through ferroelectric domain dynamics in solid-state synapses,” Nat. Commun., vol. 8, p. 14736, 2017.

[31]	C. D. Schuman et al., “A Survey of Neuromorphic Computing and Neural Networks in Hardware,” arXiv, pp. 1–88, 2017.

[32]	J. S. Seo et al., “A 45nm CMOS neuromorphic chip with a scalable architecture for learning in networks of spiking neurons,” Proc. Cust. Integr. Circuits Conf., pp. 2–5, 2011.

[33]	I. Sporea and A. Grüning, “Supervised Learning in Multilayer Spiking Neural Networks,” Neural Comput., vol. 25, no. 2, pp. 473–509, 2013.

[34]	X. Xie, H. Qu, G. Liu, M. Zhang, and J. Kurths, “An efficient supervised training algorithm for multilayer spiking neural networks,” PLoS One, vol. 11, no. 4, pp. 1–29, 2016.

[35]	C. Paper and B. C. Wiskunde, “SpikeProp : backpropagation for networks of spiking neurons . SpikeProp : Backpropagation for Networks of Spiking Neurons,” in ESANN, 8th European Symposium on Artificial Neural Networks, 2000.

[36]	F. Ponulak and A. Kasiński, “Supervised learning in spiking neural networks with ReSuMe: sequence learning, classification, and spike shifting.,” Neural Comput., vol. 22, no. 2, pp. 467–510, 2010.

[37]	P. O’Connor and M. Welling, “Deep Spiking Networks,” arXiv, pp. 1–16, 2016.

[38]	T. Shibata and T. Ohmi, “A Functional MOS Transistor Featuring Gate-Level Weighted Sum and Threshold Operations,” IEEE Trans. Electron Devices, vol. 39, no. 6, 1992.

[39]	C. Hu, “The EEPROM as an Analog Memory Device,” vol. 36, pp. 1840–1841, 1989.

[40]	L. O. Chua, “Memristor—The Missing Circuit Element,” IEEE Trans. Circuit Theory, vol. 18, no. 5, pp. 507–519, 1971.

[41]	D. B. Strukov, G. S. Snider, D. R. Stewart, and R. S. Williams, “The missing memristor found,” Nature, vol. 459, no. 7250, pp. 1154–1154, 2009.

[42]	L. Müller et al., “Electric-Field-Controlled Dopant Distribution in Organic Semiconductors,” vol. 1701466, pp. 1–7, 2017.

[43]	G. Indiveri, B. Linares-Barranco, R. Legenstein, G. Deligeorgis, and T. Prodromakis, “Integration of nanoscale memristor synapses in neuromorphic computing architectures,” Nanotechnology, vol. 24, no. 384010, 2013.

[44]	T. Prodromakis and C. Toumazou, “A review on memristive devices and applications,” 2010 IEEE Int. Conf. Electron. Circuits, Syst. ICECS 2010 - Proc., pp. 934–937, 2010.

[45]	H. Nili, S. Walia, S. Balendhran, D. B. Strukov, M. Bhaskaran, and S. Sriram, “Nanoscale resistive switching in amorphous perovskite oxide (a-SrTiO3) memristors,” Adv. Funct. Mater., vol. 24, no. 43, pp. 6741–6750, 2014.

[46]	A. C. Torrezan, J. P. Strachan, G. Medeiros-Ribeiro, and R. S. Williams, “Sub-nanosecond switching of a tantalum oxide memristor,” Nanotechnology, vol. 22, no. 48, p. 485203, 2011.

[47]	T. Berzina et al., “Optimization of an organic memristor as an adaptive memory element,” J. Appl. Phys., vol. 105, no. 12, 2009.

[48]	F. J. Pineda, “Time Dependent Adaptive Neural Networks,” Adv. Neural Inf. Process. Syst., pp. 710–718, 1990.

[49]	R. P. Feynman, “Simulating Physics with Computers,” Int. J. Theor. Phys., vol. 21, pp. 467–488, 1982.

[50]	K. Sato, D. Patterson, C. Young, and D. Patterson, “An in-depth look at Google’s first Tensor Processing Unit (TPU),” Google, 2018. [Online]. Available: https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu?fbclid=IwAR1dYdfxi4kbsC_1X31grv2ZjNUIyoBoI4Zdeh7WjVn0oIW71hf_auTMcZo. [Accessed: 29-Nov-2018].











