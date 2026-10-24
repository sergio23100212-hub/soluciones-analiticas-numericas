"""Reproducible differential-equation experiments with JSON reports."""
from __future__ import annotations
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)

def write_report(name, payload):
    payload = {'experiment': name, 'generated_utc': datetime.now(timezone.utc).isoformat(),
               'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'sympy': sp.__version__}, **payload}
    path = REPORTS / f'{name}.json'
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Saved {path}')
    return payload

def integrate_fixed(method, f, y0, t0, t_end, h):
    n = round((t_end-t0)/h)
    if n <= 0 or not math.isclose(t0+n*h, t_end, rel_tol=0, abs_tol=1e-10):
        raise ValueError('h must divide the interval exactly')
    t = np.linspace(t0,t_end,n+1)
    y = np.empty(n+1); y[0] = y0
    for i in range(n):
        ti, yi = t[i], y[i]
        if method == 'euler':
            y[i+1] = yi+h*f(ti,yi)
        elif method == 'rk4':
            k1=f(ti,yi); k2=f(ti+h/2,yi+h*k1/2)
            k3=f(ti+h/2,yi+h*k2/2); k4=f(ti+h,yi+h*k3)
            y[i+1] = yi+h*(k1+2*k2+2*k3+k4)/6
        else:
            raise ValueError(method)
    return t,y

def convergence():
    f=lambda t,y:-2*y
    rows=[]
    for method in ('euler','rk4'):
        for h in (0.2,0.1,0.05,0.025):
            start=time.perf_counter_ns()
            t,y=integrate_fixed(method,f,1.,0.,2.,h)
            elapsed=time.perf_counter_ns()-start
            exact=np.exp(-2*t)
            rows.append({'method':method,'h':h,'steps':len(t)-1,'final':float(y[-1]),
                         'exact_final':float(exact[-1]),'final_abs_error':float(abs(y[-1]-exact[-1])),
                         'max_abs_error':float(np.max(abs(y-exact))), 'elapsed_ns':elapsed})
    checks={'errors_finite':all(math.isfinite(r['max_abs_error']) for r in rows),
            'error_decreases_with_h':all(all(a['max_abs_error']>b['max_abs_error'] for a,b in zip(z,z[1:])) for z in ([r for r in rows if r['method']==m] for m in ('euler','rk4')))}
    assert all(checks.values()),checks
    return write_report('convergence',{'equation':'dy/dt=-2*y','initial':{'t':0,'y':1},'t_end':2,
                                       'rows':rows,'checks':checks,
                                       'timing_note':'Single-run timings are illustrative, not rigorous benchmarks.'})

def stability():
    A=np.array([[1.,-1.],[-1.,3.]])
    eig=np.linalg.eigvals(A)
    t=np.linspace(0,2,101)
    start=time.perf_counter_ns()
    sol=solve_ivp(lambda t,z:A@z,(0,2),[1.,1.],t_eval=t,rtol=1e-10,atol=1e-12)
    elapsed=time.perf_counter_ns()-start
    assert sol.success
    assert np.allclose(eig,[2-math.sqrt(2),2+math.sqrt(2)],rtol=1e-12)
    return write_report('stability',{'system':'dx/dt=x-y; dy/dt=-x+3y',
       'matrix':A.tolist(),'eigenvalues': [
    {
        'real': float(z.real),
        'imag': float(z.imag)
    }
    for z in sorted(eig, key=lambda z: (z.real, z.imag))
],
       'classification':'unstable node (both eigenvalues positive)',
       'initial_state':[1,1],'final_state':sol.y[:,-1].tolist(),
       'trajectory':{'t':t.tolist(),'x':sol.y[0].tolist(),'y':sol.y[1].tolist()},
       'elapsed_ns':elapsed,'checks':{'solver_success':sol.success,'positive_real_eigenvalues':bool(np.all(eig.real>0))}})

def laplace():
    t,s=sp.symbols('t s',positive=True)
    # y''+4y=sin(t), y(0)=1, y'(0)=0
    Y=sp.solve(sp.Eq((s*s+4)*sp.Symbol('Y')-s,1/(s*s+1)),sp.Symbol('Y'))[0]
    y=sp.simplify(sp.inverse_laplace_transform(Y,s,t))
    residual=sp.simplify(sp.diff(y,t,2)+4*y-sp.sin(t))
    checks={'ode_residual_zero':residual==0,'y0_one':sp.simplify(y.subs(t,0))==1,
            'dy0_zero':sp.simplify(sp.diff(y,t).subs(t,0))==0}
    assert all(checks.values()),checks
    return write_report('laplace',{'equation':'y_second+4*y=sin(t)','initial':{'y0':1,'dy0':0},
      'laplace_Y':str(Y),'solution':str(y),'checks':checks})

def main():
    results=[convergence(),stability(),laplace()]
    write_report('manifest',{'reports':[r['experiment']+'.json' for r in results],
                             'all_checks_passed':all(all(r['checks'].values()) for r in results)})

if __name__=='__main__':
    main()
