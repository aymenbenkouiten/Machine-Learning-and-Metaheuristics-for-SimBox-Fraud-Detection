@extends('layouts.webmaster')
@section('subtitle', "Call detail records")
@section('content')

@php
$user = Auth::user();
@endphp
<div class="col-12 col-lg-12 col-xxl-12 d-flex">
  <div class="card flex-fill">
    <div class="card-header d-flex justify-content-between align-items-center">
      <h5 class="card-title mb-0">Call detail records</h5>
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
                <th class="d-xl-table-cell">Djezzy number</th>
                <th class="d-xl-table-cell">Called</th>
                <th class="d-xl-table-cell">Total calls</th>
            </tr>
            </thead>
            <tbody>
            @foreach($records as $record)
            <tr>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-hashtag"></i>{{$record->id}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-barcode"></i>{{$record->Call_Type()}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-calendar"></i> {{$record->Charging_Tm}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-tape"></i> {{$record->Call_Duration}} seconds {{$record->DESTINATION_CAT}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-map"></i> {{$record->LOCATION}}<br>
                </p>
                </td>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-users"></i>{{$record->Customer()->name}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-calendar"></i>{{$record->Customer()->birthday}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-map"></i> {{$record->Customer()->birthday_location}}<br>
                </p>
                </td>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-phone-volume"></i>{{$record->MU_Device_type_Segment}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-phone-volume"></i>{{$record->generateIMEI()}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-envelope-open"></i>{{$record->generateTAC()}} {{$record->MU_HANDSET_MOBILE_TECH}} {{$record->MU_HANDSET_DUAL_SIM=="Y"?"DUAL SIM":""}}<br>
                </p>
                </td>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-phone-volume"></i>45655521<br>
                    <i class="align-middle me-2 fas fa-fw fa-envelope-open"></i>12311
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