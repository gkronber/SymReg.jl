window.BENCHMARK_DATA = {
  "lastUpdate": 1791383854337,
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
    ]
  }
}