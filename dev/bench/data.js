window.BENCHMARK_DATA = {
  "lastUpdate": 1791383856133,
  "repoUrl": "https://github.com/gkronber/SymReg.jl",
  "entries": {
    "SymRegInterpreter": [
      {
        "commit": {
          "author": {
            "email": "gabriel.kronberger@heuristiclab.com",
            "name": "Gabriel Kronberger",
            "username": "gkronber"
          },
          "committer": {
            "email": "gabriel.kronberger@heuristiclab.com",
            "name": "Gabriel Kronberger",
            "username": "gkronber"
          },
          "distinct": true,
          "id": "129b067a498da52eab618c22197bed7c6b42e4fc",
          "message": "Report coverage on Codecov and the benchmark history on GitHub Pages",
          "timestamp": "2026-10-07T16:33:46+02:00",
          "tree_id": "c4fc323d54f5ad91a2a54df16f6f22c931e5b72a",
          "url": "https://github.com/gkronber/SymReg.jl/commit/129b067a498da52eab618c22197bed7c6b42e4fc"
        },
        "date": 1791383853716,
        "tool": "julia",
        "benches": [
          {
            "name": "gradient/3-param - Float32 - 1024",
            "value": 3225,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gradient/3-param - Float32 - 16384",
            "value": 37044,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gradient/3-param - Float32 - 4096",
            "value": 13120,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gradient/3-param - Float64 - 1024",
            "value": 4767,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gradient/3-param - Float64 - 16384",
            "value": 90685,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gradient/3-param - Float64 - 4096",
            "value": 25568,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gradient/pow-log-exp - Float32 - 1024",
            "value": 79057,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gradient/pow-log-exp - Float32 - 16384",
            "value": 1443740,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gradient/pow-log-exp - Float32 - 4096",
            "value": 331001,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gradient/pow-log-exp - Float64 - 1024",
            "value": 95612,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gradient/pow-log-exp - Float64 - 16384",
            "value": 1675583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gradient/pow-log-exp - Float64 - 4096",
            "value": 390619,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/3-param - Float32 - 1024",
            "value": 1392,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/3-param - Float32 - 16384",
            "value": 12349,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/3-param - Float32 - 4096",
            "value": 4286,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/3-param - Float64 - 1024",
            "value": 1813,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/3-param - Float64 - 16384",
            "value": 24326,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/3-param - Float64 - 4096",
            "value": 8502,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/pow-log-exp - Float32 - 1024",
            "value": 71586,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/pow-log-exp - Float32 - 16384",
            "value": 1204685,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/pow-log-exp - Float32 - 4096",
            "value": 276550,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/pow-log-exp - Float64 - 1024",
            "value": 79548,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/pow-log-exp - Float64 - 16384",
            "value": 1383470,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/pow-log-exp - Float64 - 4096",
            "value": 380433,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/x²+y² - Float32 - 1024",
            "value": 1022,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/x²+y² - Float32 - 16384",
            "value": 7301,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/x²+y² - Float32 - 4096",
            "value": 2954,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/x²+y² - Float64 - 1024",
            "value": 1472,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/x²+y² - Float64 - 16384",
            "value": 20811,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "interpret/x²+y² - Float64 - 4096",
            "value": 4237,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/3-param - Float32 - 1024",
            "value": 4106,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/3-param - Float32 - 16384",
            "value": 40980,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/3-param - Float32 - 4096",
            "value": 14021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/3-param - Float64 - 1024",
            "value": 5297,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/3-param - Float64 - 16384",
            "value": 83644,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/3-param - Float64 - 4096",
            "value": 20420,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/pow-log-exp - Float32 - 1024",
            "value": 81020,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/pow-log-exp - Float32 - 16384",
            "value": 1503037,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/pow-log-exp - Float32 - 4096",
            "value": 339503,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/pow-log-exp - Float64 - 1024",
            "value": 97394,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/pow-log-exp - Float64 - 16384",
            "value": 1713619,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/pow-log-exp - Float64 - 4096",
            "value": 439311,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/x²+y² - Float32 - 1024",
            "value": 1583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/x²+y² - Float32 - 16384",
            "value": 14942,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/x²+y² - Float32 - 4096",
            "value": 5337,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/x²+y² - Float64 - 1024",
            "value": 2364,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/x²+y² - Float64 - 16384",
            "value": 28633,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "jacobian/x²+y² - Float64 - 4096",
            "value": 7361,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      }
    ],
    "SymRegLikelihoods": [
      {
        "commit": {
          "author": {
            "email": "gabriel.kronberger@heuristiclab.com",
            "name": "Gabriel Kronberger",
            "username": "gkronber"
          },
          "committer": {
            "email": "gabriel.kronberger@heuristiclab.com",
            "name": "Gabriel Kronberger",
            "username": "gkronber"
          },
          "distinct": true,
          "id": "129b067a498da52eab618c22197bed7c6b42e4fc",
          "message": "Report coverage on Codecov and the benchmark history on GitHub Pages",
          "timestamp": "2026-10-07T16:33:46+02:00",
          "tree_id": "c4fc323d54f5ad91a2a54df16f6f22c931e5b72a",
          "url": "https://github.com/gkronber/SymReg.jl/commit/129b067a498da52eab618c22197bed7c6b42e4fc"
        },
        "date": 1791383855680,
        "tool": "julia",
        "benches": [
          {
            "name": "cosmic/nll/Float64 / 1024 obs",
            "value": 6632,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48\nallocs=1\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "cosmic/nll/Float64 / 4096 obs",
            "value": 21790,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48\nallocs=1\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "cosmic/nll + grad/Float64 / 1024 obs",
            "value": 9878,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48\nallocs=1\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "cosmic/nll + grad/Float64 / 4096 obs",
            "value": 37980,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48\nallocs=1\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "cosmic/nll + grad fd/Float64 / 1024 obs",
            "value": 21390,
            "unit": "ns",
            "extra": "gctime=0\nmemory=352\nallocs=7\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "cosmic/nll + grad fd/Float64 / 4096 obs",
            "value": 81302,
            "unit": "ns",
            "extra": "gctime=0\nmemory=352\nallocs=7\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "fim/gaussian known/Float64 / 1024 obs",
            "value": 60312,
            "unit": "ns",
            "extra": "gctime=0\nmemory=848\nallocs=13\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "fim/gaussian profiled/Float64 / 1024 obs",
            "value": 81212,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1104\nallocs=13\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "fim/gaussian scatter/Float64 / 1024 obs",
            "value": 99826,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1104\nallocs=13\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "fim/laplace profiled/Float64 / 1024 obs",
            "value": 20979,
            "unit": "ns",
            "extra": "gctime=0\nmemory=304\nallocs=4\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "fim/mnr diagonal/Float64 / 1024 obs",
            "value": 428980,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5376\nallocs=61\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll/Float32 / 1024 obs",
            "value": 1844,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll/Float32 / 4096 obs",
            "value": 5951,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll/Float64 / 1024 obs",
            "value": 2144,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll/Float64 / 4096 obs",
            "value": 8236,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll + grad/Float32 / 1024 obs",
            "value": 3767,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll + grad/Float32 / 4096 obs",
            "value": 13225,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll + grad/Float64 / 1024 obs",
            "value": 4990,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll + grad/Float64 / 4096 obs",
            "value": 24666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll + grad fd/Float32 / 1024 obs",
            "value": 10219,
            "unit": "ns",
            "extra": "gctime=0\nmemory=240\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll + grad fd/Float32 / 4096 obs",
            "value": 36098,
            "unit": "ns",
            "extra": "gctime=0\nmemory=240\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll + grad fd/Float64 / 1024 obs",
            "value": 12493,
            "unit": "ns",
            "extra": "gctime=0\nmemory=288\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/nll + grad fd/Float64 / 4096 obs",
            "value": 41678,
            "unit": "ns",
            "extra": "gctime=0\nmemory=288\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/optimize LM/Float32 / 1024 obs",
            "value": 30034,
            "unit": "ns",
            "extra": "gctime=0\nmemory=224\nallocs=6\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/optimize LM/Float32 / 4096 obs",
            "value": 101611.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=224\nallocs=6\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/optimize LM/Float64 / 1024 obs",
            "value": 31452.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=240\nallocs=6\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/optimize LM/Float64 / 4096 obs",
            "value": 129966.4,
            "unit": "ns",
            "extra": "gctime=0\nmemory=240\nallocs=6\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/optimize NLopt/Float64 / 1024 obs",
            "value": 57356.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=67936\nallocs=70\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "gaussian/optimize NLopt/Float64 / 4096 obs",
            "value": 197532.4,
            "unit": "ns",
            "extra": "gctime=0\nmemory=264544\nallocs=70\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/cosmic nll/Float64 / 1024 obs",
            "value": 12413,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48\nallocs=1\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/cosmic nll + grad/Float64 / 1024 obs",
            "value": 17773,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48\nallocs=1\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/gaussian nll/Float64 / 1024 obs",
            "value": 10349,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/gaussian nll + grad/Float64 / 1024 obs",
            "value": 12934,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/laplace nll/Float64 / 1024 obs",
            "value": 11241,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/laplace nll + grad/Float64 / 1024 obs",
            "value": 16200,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/mnr diagonal off nll/Float64 / 1024 obs",
            "value": 63719,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/mnr diagonal off nll + grad/Float64 / 1024 obs",
            "value": 78978,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3248\nallocs=39\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/xprofile dense nll/Float64 / 64 obs",
            "value": 40826,
            "unit": "ns",
            "extra": "gctime=0\nmemory=576\nallocs=2\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/xprofile dense nll + grad/Float64 / 64 obs",
            "value": 48180,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=44\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/xuniform diagonal nll/Float64 / 1024 obs",
            "value": 177471,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "intrinsic scatter/xuniform diagonal nll + grad/Float64 / 1024 obs",
            "value": 366363,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll/Float32 / 1024 obs",
            "value": 2043,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll/Float32 / 4096 obs",
            "value": 6382,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll/Float64 / 1024 obs",
            "value": 2335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll/Float64 / 4096 obs",
            "value": 9057,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll (profiled scale)/Float32 / 1024 obs",
            "value": 1913,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll (profiled scale)/Float32 / 4096 obs",
            "value": 6222,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll (profiled scale)/Float64 / 1024 obs",
            "value": 2305,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll (profiled scale)/Float64 / 4096 obs",
            "value": 8216,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll + grad/Float32 / 1024 obs",
            "value": 3917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll + grad/Float32 / 4096 obs",
            "value": 13836,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll + grad/Float64 / 1024 obs",
            "value": 5591,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll + grad/Float64 / 4096 obs",
            "value": 20418,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll + grad fd/Float32 / 1024 obs",
            "value": 10770,
            "unit": "ns",
            "extra": "gctime=0\nmemory=240\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll + grad fd/Float32 / 4096 obs",
            "value": 40725,
            "unit": "ns",
            "extra": "gctime=0\nmemory=240\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll + grad fd/Float64 / 1024 obs",
            "value": 11852,
            "unit": "ns",
            "extra": "gctime=0\nmemory=288\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/nll + grad fd/Float64 / 4096 obs",
            "value": 44914,
            "unit": "ns",
            "extra": "gctime=0\nmemory=288\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/optimize NLopt/Float64 / 1024 obs",
            "value": 192753.4,
            "unit": "ns",
            "extra": "gctime=0\nmemory=704\nallocs=18\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "laplace/optimize NLopt/Float64 / 4096 obs",
            "value": 841428.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=704\nallocs=18\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr banded/nll/Float64 / 256 obs",
            "value": 19948,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr banded/nll/Float64 / 64 obs",
            "value": 6302,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr banded/nll + grad/Float64 / 256 obs",
            "value": 33503,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5312\nallocs=51\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr banded/nll + grad/Float64 / 64 obs",
            "value": 12634,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5312\nallocs=51\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr banded/nll + grad fd/Float64 / 256 obs",
            "value": 112169,
            "unit": "ns",
            "extra": "gctime=0\nmemory=672\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr banded/nll + grad fd/Float64 / 64 obs",
            "value": 35556,
            "unit": "ns",
            "extra": "gctime=0\nmemory=672\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr dense/nll/Float64 / 256 obs",
            "value": 7910286,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6288\nallocs=6\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr dense/nll/Float64 / 64 obs",
            "value": 369098,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1696\nallocs=4\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr dense/nll + grad/Float64 / 256 obs",
            "value": 746098343,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20208\nallocs=33\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr dense/nll + grad/Float64 / 64 obs",
            "value": 12241101,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6432\nallocs=27\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr dense/nll + grad fd/Float64 / 256 obs",
            "value": 752431894,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20208\nallocs=33\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr dense/nll + grad fd/Float64 / 64 obs",
            "value": 12149511,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6432\nallocs=27\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr diagonal/nll/Float64 / 1024 obs",
            "value": 62416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr diagonal/nll/Float64 / 4096 obs",
            "value": 259083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr diagonal/nll + grad/Float64 / 1024 obs",
            "value": 78496,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3248\nallocs=39\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr diagonal/nll + grad/Float64 / 4096 obs",
            "value": 309277,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3920\nallocs=51\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr diagonal/nll + grad fd/Float64 / 1024 obs",
            "value": 431144,
            "unit": "ns",
            "extra": "gctime=0\nmemory=448\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "mnr diagonal/nll + grad fd/Float64 / 4096 obs",
            "value": 1488595,
            "unit": "ns",
            "extra": "gctime=0\nmemory=448\nallocs=5\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian profiled fim/Float32 / 1024 obs",
            "value": 453365,
            "unit": "ns",
            "extra": "gctime=0\nmemory=912\nallocs=14\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian profiled fim/Float64 / 1024 obs",
            "value": 509891,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=14\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian profiled nll/Float32 / 1024 obs",
            "value": 68077,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian profiled nll/Float64 / 1024 obs",
            "value": 80350,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian profiled nll + grad/Float32 / 1024 obs",
            "value": 147294,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian profiled nll + grad/Float64 / 1024 obs",
            "value": 181969,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian profiled optimize LM/Float32 / 1024 obs",
            "value": 1637895.4,
            "unit": "ns",
            "extra": "gctime=0\nmemory=304\nallocs=6\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian profiled optimize LM/Float64 / 1024 obs",
            "value": 1348506,
            "unit": "ns",
            "extra": "gctime=0\nmemory=320\nallocs=6\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian scatter fim/Float32 / 1024 obs",
            "value": 485926,
            "unit": "ns",
            "extra": "gctime=0\nmemory=912\nallocs=14\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian scatter fim/Float64 / 1024 obs",
            "value": 559463,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=14\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian scatter nll + grad/Float32 / 1024 obs",
            "value": 154528,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian scatter nll + grad/Float64 / 1024 obs",
            "value": 185566,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian scatter optimize NLopt/Float32 / 1024 obs",
            "value": 691563.4,
            "unit": "ns",
            "extra": "gctime=0\nmemory=736\nallocs=18\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "pow-log-exp/gaussian scatter optimize NLopt/Float64 / 1024 obs",
            "value": 912393.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=18\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xprofile dense/nll/Float64 / 256 obs",
            "value": 778993,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2120\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xprofile dense/nll/Float64 / 64 obs",
            "value": 43241,
            "unit": "ns",
            "extra": "gctime=0\nmemory=576\nallocs=2\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xprofile dense/nll + grad/Float64 / 256 obs",
            "value": 810301,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7560\nallocs=45\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xprofile dense/nll + grad/Float64 / 64 obs",
            "value": 48701,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6016\nallocs=44\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xprofile dense/nll + grad fd/Float64 / 256 obs",
            "value": 11317671,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4560\nallocs=11\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xprofile dense/nll + grad fd/Float64 / 64 obs",
            "value": 260205,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1472\nallocs=9\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xuniform diagonal/nll/Float64 / 1024 obs",
            "value": 179425,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xuniform diagonal/nll/Float64 / 4096 obs",
            "value": 721616,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xuniform diagonal/nll + grad/Float64 / 1024 obs",
            "value": 361073,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xuniform diagonal/nll + grad/Float64 / 4096 obs",
            "value": 1545782,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xuniform diagonal/nll + grad fd/Float64 / 1024 obs",
            "value": 425223,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13600\nallocs=83\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "xuniform diagonal/nll + grad fd/Float64 / 4096 obs",
            "value": 1729694,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13600\nallocs=83\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":5,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      }
    ]
  }
}