// Package adf implements the Augmented Dickey-Fuller test. This is a port of
// the implementation of adf given here:
// https://github.com/Netflix/Surus/blob/master/src/main/java/org/surus/math/AugmentedDickeyFuller.java.
package adf

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"

	"math"
)

const (
	LPenalty      = 0.0001 // L penalty to pass to ridge regression
	DefaultPValue = -3.45  // Test p-value threshold
)

// An instance of an ADF test
type ADF struct {
	Series          []float64 // The time series to test
	PValueThreshold float64   // The p-value threshold for the test
	Statistic       float64   // The test statistic
	Lag             int       // The lag to use when running the test
}

// New creates and returns a new ADF test.
func NewADF(series []float64, pvalue float64, lag int) *ADF {
	if pvalue == 0 {
		pvalue = DefaultPValue
	}

	if lag < 0 {
		lag = int(math.Floor(math.Cbrt(float64(len(series)))))
	}

	newSeries := make([]float64, len(series))
	copy(newSeries, series)

	return &ADF{Series: newSeries, PValueThreshold: pvalue, Lag: lag}
}

// Run runs the Augmented Dickey-Fuller test.
func (adf *ADF) Run() {
	series := adf.Series
	mean := stat.Mean(series, nil)

	if mean != 0.0 {
		for i, v := range series {
			series[i] = v - mean
		}
	}

	n := len(series) - 1
	y := diff(series)
	lag := adf.Lag
	k := lag + 1

	z := laggedMatrix(y, k)

	zcol1 := mat.Col(nil, 0, z)
	xt1 := series[k-1 : n]
	r, c := z.Dims()

	var design *mat.Dense

	if k > 1 {
		yt1 := view(z, 0, 1, r, c-1)
		design = mat.NewDense(n-k+1, k, nil)
		design.SetCol(0, xt1)

		_, c = yt1.Dims()

		for i := 0; i < c; i++ {
			design.SetCol(1+i, mat.Col(nil, i, yt1))
		}

	} else {
		design = mat.NewDense(n-k+1, 1, nil)
		design.SetCol(0, xt1)
	}

	regressY := mat.NewVecDense(len(zcol1), zcol1)

	rr := NewRidge(design, regressY, LPenalty)
	rr.Regress()

	beta := rr.Coefficients.RawVector().Data
	sd := rr.StdErrs

	adf.Statistic = beta[0] / sd[0]
}

// IsStationary returns true if the tested time series is stationary.
func (adf ADF) IsStationary() bool {
	return adf.Statistic < adf.PValueThreshold
}

func diff(x []float64) []float64 {
	y := make([]float64, len(x)-1)
	for i := 0; i < len(x)-1; i++ {
		y[i] = x[i+1] - x[i]
	}
	return y
}

func laggedMatrix(series []float64, lag int) *mat.Dense {
	r, c := len(series)-lag+1, lag
	m := mat.NewDense(r, c, nil)

	for j := 0; j < c; j++ {
		for i := 0; i < r; i++ {
			m.Set(i, j, series[lag-j-1+i])
		}
	}
	return m
}

func view(m *mat.Dense, i, j, r, c int) mat.Matrix {
	return m.Slice(i, i+r, j, j+c)
}
