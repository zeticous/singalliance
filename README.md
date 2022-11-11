# Introduction

The following code is used for submission of the SingAlliance coding challenge.

# How to Run

Before running the script, make sure all the requirements are met. If not, made sure `pip` is installed, and run the following:
> pip install -r requirements.txt

The script can be executed using the following command:
> python3 main.py

The global minimum variance portfolio (GMVP) is computed in closed form. The efficient frontier is then computed using the 
two-fund theorem, where a linear combination of the GMVP and the maximum return asset can be used to form the frontier.

The minimum variance portfolio is found in `gmvp.txt` and the efficient frontier is found in `frontier.png`.