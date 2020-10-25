# Rolade

Minimal Neural Network toolkit.

## Before you go too far

Ini cuma project hore-hore saja. Tidak ada jaminan atas validitas dari metode-metode yang digunakan, atau adanya support maupun pengembangan lebih jauh. Ada baiknya untuk tidak menggunakan pada project yang serius. Semua kerugian yang ditimbulkan akibat menggunakan toolkit ini di luar tanggung jawab kami.

## Getting Started

Berikut adalah langkah-langkah untuk menggunakan toolkit ini. Mungkin kurang lengkap, tapi ndak apa-apa. 

### Defining Network

Untuk memulai, pertama import dulu networknya.

```
import "github.com/harungurubudi/rolade/network"
```

Kemudian definisikan struktur network yang anda inginkan. Misal, menginginkan network dengan feature berukuran ``4`` dan target berukuran ``2`` :

```
nt := network.NewNetwork(4, 2)
```

Tentukan juga hidden layer yang diinginkan. Untuk menambahkan hidden layer, bisa menggunakan method ``AddHiddenLayer`` pada object network. Misal, anda ingin menambahkan dua hidden layer dengan ukuran masing-masing ``4`` dan ``3`` : 

```
nt.AddHiddenLayer(4)
nt.AddHiddenLayer(3)
```


### Network Properties

Seperti kebanyakan toolkit yang lain, biar keliatan customable, toolkit ini juga bisa diset properti-propertinya sesuai dengan kebutuhan. 

```
nt.SetProps(network.Props{
    Optimizer: &optimizer.SGD{
        Alpha: 0.01,
    },
    ErrLimit: 0.005,
    MaxEpoch: 10000,
})
```

Berikut daftar properties yang tersedia : 

| Options       | Description                                       | Type                      | Default Value           |
|---------------|---------------------------------------------------|---------------------------|-------------------------|
| Activation    | Fungsi Aktivasi                                   | activation.IActivation    | *activation.Tanh        |
| Optimizer     | Algoritma Optimizer                               | optimizer.IOptimizer      | *optimizer.SGD          |
| Loss          | Fungsi Loss                                       | loss.ILoss                | *loss.RMSE              |
| ErrLimit      | Batas error yang perlu dicapai saat training      | float64                   | 0.001                   |
| MaxEpoch      | Epoch maksimal saat training                      | int                       | 1000                    |


### Datatype

Semua data yang digunakan baik untuk input maupun output dalam toolkit ini adalah data ber tipe ``network.DataArray`` yang merupakan tipe kustom dari ``[]float64``. Untuk mendeklarasikan data tersebut, dapat dilihat contoh berikut : 

```
singleData := network.DataArray{0, 0, 0, 1}
```


### Training

Untuk melakukan proses network training, kita dapat menggunakan method ``Train`` pada network : 

```
nt.Train(inputs []DataArray, targets []DataArray) (error)
``` 

Dengan argument : 
1. ``inputs``, set data input. Dengan type array of ``network.DataArray``.
2. ``targets``, set data target. Dengan type array of ``network.DataArray``.


### Testing

Untuk melakukan test dari network, kita dapat menggunakan method ``Test`` pada network : 

```
nt.Test(input DataArray) (DataArray, []int, error)
```
Dengan argument : 
1. ``inputs``, set data input. Dengan type array of ``network.DataArray``.

Method ini mengembalikan tiga result :
1. Set output asli dari proses testing. Bertipe ``network.DataArray``.
2. Nilai pembulatan dari output testing. Bertipe ``[]int``. Jika output asli > 0.5 maka akan menjadi 1, sebaliknya akan menjadi 0.
3. Objek error jika proses testing gagal.


## Activation Function
Pada toolkit ini, tersedia tiga activation function yang dapat digunakan dalam properties : 
1. Sigmoid : *activation.Sigmoid
2. Tanh : *activation.Tanh
3. RELU : *activation.Relu


## Optimizer
Berikut optimizer algorithm yang tersedia juga. Cuma satu sih :(  

1. SGD : *optimizer.SGD. Dengan properties sebagai berikut :
    - ``Alpha`` (learning rates) : float64
    - ``M`` (momentum) : float64
    - ``IsNesterov`` (Nesterov toggle) : bool
    - ``V`` (Velocity) : float64


## Loss
Sedangkan berikut loss function yang tersedia. Cuma satu juga :(( 

1.RMSE : *loss.RMSE


## Minimum Example

Contoh sederhana bisa dilihat di directory example.

## To Do 
1. Save and load model
2. Batch paralel training
