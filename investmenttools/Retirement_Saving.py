from numpy import *
import pylab as py


'''
Assumptions in the analysis below:
All cash flow occurs at end of the year
Returns and Inflation rate are assumed to be certain and fixed
Annual growth in savings is assumed to be constant and the same
No assumption of money moving into safer accounts over time.. this has impact on rate of return
Withdrawal dollars in terms of todays value assume that we know how many years precisely we want all our money to last
'''

def Retirement_Savings(savings_per_year=1000,savings_growth_rate=5,years_to_retire=25,rate_of_return=5):
  Vn=0
  c=savings_per_year
  g=savings_growth_rate
  r=rate_of_return
  n=years_to_retire
  for i in range(n):
    Vn+=c*((1+0.01*g)**i)*(1+0.01*r)**(n-(i+1))
  return Vn

def Retirement_Withdrawals(retirement_savings=100000,withdrawal_rate=3,rate_of_return=5,years_money_lasting=25,years_to_retire=35):
  w=withdrawal_rate
  r=rate_of_return
  s=retirement_savings
  n=years_money_lasting
  v=retirement_savings
  yr=years_to_retire
  if w==r:
    c=v*(1+0.01*r)/n
  else:
    c=0.01*v*(r-w)/(1-((1+0.01*w)/(1+0.01*r))**n)
  c_today=c/(1+0.01*w)**yr
  return c_today