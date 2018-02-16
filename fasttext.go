package fasttextgo

// #cgo LDFLAGS: -L${SRCDIR} -lfasttext -lstdc++
// #include <stdlib.h>
// void load_model(char *path);
// int predict(char *query, int k, float* prob, char** buf, int buf_sz);
import "C"
import (
	"errors"
	"unsafe"

	"github.com/chewxy/math32"
)

type Tag struct {
	Label       string
	Probability float32
}

// LoadModel - load FastText model
func LoadModel(path string) {
	C.load_model(C.CString(path))
}

// Predict - predict
func Predict(sentence string, k int) ([]Tag, error) {

	bufs := make([]*C.char, k)
	for i, _ := range bufs {
		bufs[i] = (*C.char)(C.malloc(C.size_t(64)))
	}
	probs := make([]C.float, k)

	defer func() {
		for i, _ := range bufs {
			C.free(unsafe.Pointer(bufs[i]))
		}
	}()

	ret := C.predict(C.CString(sentence), C.int(k), &probs[0], &bufs[0], 64)
	if ret == 0 {
		fs := make([]float32, k)
		for i, p := range probs {
			fs[i] = math32.Exp(float32(p))
		}
		var tags []Tag
		for i, _ := range bufs {
			tags = append(tags, Tag{
				Label:       C.GoString(bufs[i]),
				Probability: float32(fs[i]),
			})
		}
		return tags, nil
	}
	return nil, errors.New("error in prediction")

}
