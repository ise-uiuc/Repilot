# Repilot Patch

```
clone.classNames = null;
```

# Developer Patch

```
        clone.classNames = null; // derived on first hit, otherwise gets a pointer to source classnames
```

# Context

```
--- bug/Jsoup-32/src/main/java/org/jsoup/nodes/Element.java

+++ fix/Jsoup-32/src/main/java/org/jsoup/nodes/Element.java

@@ -1135,7 +1135,7 @@

     @Override
     public Element clone() {
         Element clone = (Element) super.clone();
-        clone.classNames();
+clone.classNames = null;
         return clone;
     }
 }
```

# Note

