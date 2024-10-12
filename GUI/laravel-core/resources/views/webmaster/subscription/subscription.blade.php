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
                <th class="d-xl-table-cell">Information</th>
                <th class="d-xl-table-cell">Established by</th>
                <th class="d-xl-table-cell">Total calls</th>
            </tr>
            </thead>
            <tbody>
              @foreach($subscriptions as $subscription)
            <tr>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-hashtag"></i>{{$subscription->id}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-barcode"></i>{{$subscription->phone_number}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-calendar"></i> {{$subscription->created_at}}<br>
                </p>
                </td>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-users"></i>{{$subscription->Customer()->name}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-calendar"></i>{{$subscription->Customer()->birthday}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-map"></i> {{$subscription->Customer()->birthday_location}}<br>
                </p>
                </td>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-map"></i> {{$subscription->established_by}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-calendar"></i> {{$subscription->created_at}}<br>
                </p>
                </td>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-phone-volume"></i>{{$subscription->Calls()}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-envelope-open"></i>{{$subscription->SMS()}}
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