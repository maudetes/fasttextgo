package fasttextgo

import (
	"fmt"
	"testing"
)

func TestSimple(t *testing.T) {
	LoadModel("/home/estelle/go/src/github.com/cozy/cozy-stack/pkg/index/lid.176.ftz")
	tags, err := PredictK("我是法国人", 6)
	if err != nil {
		panic(err)
	}
	for _, t := range tags {
		fmt.Println(t.Label, t.Prob)
	}
	// fmt.Println(Predict("sushi in palo alto"))
	// fmt.Println(Predict("muy caliente"))
}
