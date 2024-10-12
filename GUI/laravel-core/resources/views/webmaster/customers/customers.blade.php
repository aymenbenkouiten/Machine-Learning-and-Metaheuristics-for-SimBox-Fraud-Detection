@extends('layouts.webmaster')
@section('subtitle', "Customers")
@section('content')

@php
$user = Auth::user();
@endphp
<div class="col-12 col-lg-12 col-xxl-12 d-flex">
  <div class="card flex-fill">
    <div class="card-header d-flex justify-content-between align-items-center">
      <h5 class="card-title mb-0">Customers</h5>
    </div>
  </div>
</div>
<div class="col-12 col-lg-12 col-xxl-12 d-flex">
  <div class="card flex-fill">
	<div class="card-body">
      <div class="table-responsive">
        <table class="table table-hover my-0" id="customers">
            <thead>
            <tr>
                <th class="d-xl-table-cell">ID</th>
                <th class="d-xl-table-cell">Customer</th>
                <th class="d-xl-table-cell">Numbers</th>
                <th class="d-xl-table-cell">Total calls</th>
            </tr>
            </thead>
            <tbody>
            @foreach($customers as $customer)
            <tr>
                <td class="d-xl-table-cell single-line">
                  <p>
                      <i class="align-middle me-2 fas fa-fw fa-hashtag"></i>{{$customer->id}}<br>
                      <i class="align-middle me-2 fas fa-fw fa-barcode"></i>{{$customer->NIN}}<br>
                      <i class="align-middle me-2 fas fa-fw fa-calendar"></i> {{$customer->created_at}}<br>
                  </p>
                </td>
                <td class="d-xl-table-cell single-line">
                  <p>
                      <i class="align-middle me-2 fas fa-fw fa-users"></i>{{$customer->name}}<br>
                      <i class="align-middle me-2 fas fa-fw fa-calendar"></i>{{$customer->birthday}}<br>
                      <i class="align-middle me-2 fas fa-fw fa-map"></i> {{$customer->birthday_location}}<br>
                  </p>
                </td>
                <td class="d-xl-table-cell single-line">
                <p>
                  @foreach($customer->Phones() as $phone)
                    <i class="align-middle me-2 fas fa-fw fa-phone"></i>{{$phone->phone_number}}<br>
                  @endforeach
                </p>
                </td>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-phone-volume"></i>{{$phone->Calls()}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-envelope-open"></i>{{$phone->SMS()}}
                </p>
                </td>

            </tr>
            @endforeach
            </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
@endsection
@section('script')
<script src="{{asset('assets/js/datatables.js')}}"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    $("#customers").DataTable({
      responsive: false
    });
  });
</script>
@endsection