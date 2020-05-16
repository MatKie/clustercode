Example by: T Lindeboom
The example is of a similation of water and a monoglyceride (referred MGE C18) 
using a coarse grain model by T Lindeboom. The simulation is in a triclinic box 
and starts in a mixed state and equilibrates to an laminar phase.

Simulation details:
    - Time: 500 ns
    - Temperature: 423 K
    - Water mass fraction: 0.25
    - Pressure: 10 bar
    - Triclinic box
    - Configurations output frequency: every 10 ns

Monoglyceride
    - name: MGE
    - number: 4000
    - atoms and counts:
        GLY 2
        ES 1
        CM 5
        CE 1
Water (each coarse grain segment represents two water molecules)
    - name: SOL
    - number: 13713
    - atoms and counts:
        W2 1