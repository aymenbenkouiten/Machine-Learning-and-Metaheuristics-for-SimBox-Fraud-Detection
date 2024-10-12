<?php
return array(
    array("type" => "text", "content"=> "General", "permissions" => ["dashboard_consult"]),
    array(
        "type" => "link",
        "content" => "Dashboard",
        "permissions" => "dashboard_consult",
        "section" => "dashboard",
        "route" => "dashboard",
        "icon" => array("type" => "feather", "content" => "pie-chart"),
    ),
    array("type" => "text", "content"=> "Customers", "permissions" => ["subscriptions_consult", "customers_consult", "cdr_consult"]),
    array(
        "type" => "link",
        "content" => "Customers",
        "permissions" => "customers_consult",
        "section" => "customers",
        "route" => "customers",
        "icon" => array("type" => "feather", "content" => "users"),
    ),
    array(
        "type" => "link",
        "content" => "Subscriptions",
        "permissions" => "subscriptions_consult",
        "section" => "subscriptions",
        "route" => "subscriptions",
        "icon" => array("type" => "feather", "content" => "settings"),
    ),
    array(
        "type" => "link",
        "content" => "Call detail records",
        "permissions" => "cdr_consult",
        "section" => "cdr",
        "route" => "cdr",
        "icon" => array("type" => "feather", "content" => "phone"),
    ),
    array("type" => "text", "content"=> "Fraud detection", "permissions" => ["suspicious_consult", "barred_consult"]),
    array(
        "type" => "link",
        "content" => "Suspicious activities",
        "permissions" => "suspicious_consult",
        "section" => "fraudssuspicious",
        "route" => "fraudssuspicious",
        "icon" => array("type" => "feather", "content" => "crosshair"),
    ),
    array(
        "type" => "link",
        "content" => "Barred numbers",
        "permissions" => "barred_consult",
        "section" => "fraudsbarred",
        "route" => "fraudsbarred",
        "icon" => array("type" => "feather", "content" => "power"),
    ),
    
    array("type" => "text", "content"=> "Administration", "permissions" => ["users_consult", "settings_consult"]),
    array(
        "type" => "link",
        "content" => "Users",
        "permissions" => "users_consult",
        "section" => "users",
        "route" => "users",
        "icon" => array("type" => "feather", "content" => "user"),
    ),
    array(
        "type" => "link",
        "content" => "Settings",
        "permissions" => "settings_consult",
        "section" => "settings",
        "route" => "settings",
        "icon" => array("type" => "feather", "content" => "settings"),
    ),

    
);