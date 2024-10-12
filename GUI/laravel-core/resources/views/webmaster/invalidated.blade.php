@extends('layouts.webmaster')
@section('subtitle', "Invalidated user")
@section('content')

@php
$user = Auth::user();
@endphp

@endsection