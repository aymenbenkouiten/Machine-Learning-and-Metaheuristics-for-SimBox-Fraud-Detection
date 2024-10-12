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
                <th class="d-xl-table-cell">Decision</th>
            </tr>
            </thead>
            <tbody>
            @foreach($suspiciouses as $suspicious)
            <tr>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-hashtag"></i>{{$suspicious->id}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-barcode"></i>{{$suspicious->Subscription()->phone_number}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-map"></i> {{$suspicious->Subscription()->established_by}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-calendar"></i> {{$suspicious->created_at}}<br>
                </p>
                </td>
                <td class="d-xl-table-cell single-line">
                  <p>
                      <i class="align-middle me-2 fas fa-fw fa-users"></i>{{$suspicious->Customer()->name}}<br>
                      <i class="align-middle me-2 fas fa-fw fa-calendar"></i>{{$suspicious->Customer()->birthday}}<br>
                      <i class="align-middle me-2 fas fa-fw fa-map"></i> {{$suspicious->Customer()->birthday_location}}<br>
                  </p>
                </td>
                <td class="d-xl-table-cell single-line">
                <p>
                  @foreach($suspicious->Customer()->Phones() as $phone)
                    <i class="align-middle me-2 fas fa-fw fa-phone"></i>{{$phone->phone_number}}<br>
                  @endforeach
                </p>
                </td>
                @if($user->Has_Permission('suspicious_check'))
                <td class="d-xl-table-cell single-line">
                    <button data-bs-toggle="modal" data-bs-target="#notfraud{{$suspicious->id}}" class="btn btn-success">Not fraud</button>
                    <button data-bs-toggle="modal" data-bs-target="#fraud{{$suspicious->id}}" class="btn btn-danger">Fraud</button>
                </td>
                @endif
            </tr>
            @endforeach
            </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
@if($user->Has_Permission('suspicious_check'))
@foreach($suspiciouses as $suspicious)
<div class="modal fade" id="fraud{{$suspicious->id}}" tabindex="-1" role="dialog" aria-hidden="true">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <form action="{{route('fraudsfraud', $suspicious->id)}}" method="POST">
        @csrf
        @method('put')
        <div class="modal-header">
            <h5 class="modal-title">You sure to ban {{$suspicious->Customer()->name}} ?</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
            <button type="submit" class="btn btn-danger">Ban</button>
        </div>
      </form>
    </div>
  </div>
</div>
<div class="modal fade" id="notfraud{{$suspicious->id}}" tabindex="-1" role="dialog" aria-hidden="true">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <form action="{{route('fraudsnotfraud', $suspicious->id)}}" method="POST">
        @csrf
        @method('put')
        <div class="modal-header">
            <h5 class="modal-title">You sure to unban {{$suspicious->Customer()->name}} ?</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
            <button type="submit" class="btn btn-success">Not Fraud</button>
        </div>
      </form>
    </div>
  </div>
</div>
@endforeach
@endif
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